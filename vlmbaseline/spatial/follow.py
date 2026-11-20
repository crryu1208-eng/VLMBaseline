"""Guidance while walking: given a goal, feed poses as the user walks and get short Korean announcements.

    f = Follower(idx, goal="O365")            # or label="potted plant", or area="엘리베이터 앞"
    f.start(pose)                              # plans the route from this pose
    for pose in stream:                        # e.g. 2 Hz from the mapping pipeline
        r = f.update(pose)                     # {'say': [...], 'next': {...}, 'remaining_m', 'arrived', ...}

Everything is computed in code (no VLM), so one update takes about a millisecond.
The route is the walking path over the place graph cut into legs (nav.legs); the user's position is projected
on it with progress that only moves forward (a ring corridor must not make the user jump to the other side).
Announcements, each once: the first leg at start, 'N m 앞에서 <turn>' when a corner is ahead, '<turn>' at the
corner, '반대 방향' when walking away from the route, a new route after leaving it, and arrival with the side of
the goal.
Leaving the route: more than off_m from the route line AND the walk to the goal from here (place graph) is more than
detour_m above the best so far, or 2 x detour_m above what is left of the plan (two updates in a row). Walking beside
the line in an open room, or reaching the goal from another side, keeps getting closer and is not leaving.
"""
from __future__ import annotations

import math
import re

import numpy as np

from vlmbaseline.spatial import nav
from vlmbaseline.spatial.names import KO
from vlmbaseline.spatial.tools import SKIP_LANDMARK

SAY_TURN = {"직진": "계속 직진하세요", "왼쪽으로 비스듬히": "왼쪽으로 비스듬히 가세요",
            "오른쪽으로 비스듬히": "오른쪽으로 비스듬히 가세요", "왼쪽으로 돌기": "왼쪽으로 도세요",
            "오른쪽으로 돌기": "오른쪽으로 도세요", "뒤로 돌기": "뒤로 돌아서 가세요"}


def josa(word, pair=("은", "는")):
    """word + particle by the final consonant of its last Korean syllable; a trailing '(O12)' is skipped,
    so '벤치(O223)' takes the particle of '벤치'. pair = (after a consonant, after a vowel); for ('으로', '로')
    a final ㄹ counts as a vowel."""
    base = re.sub(r"\s*\([^)]*\)\s*$", "", word or "").rstrip()
    ch = base[-1:]
    if not ("가" <= ch <= "힣"):
        return f"{word}{pair[0]}({pair[1]})"
    jong = (ord(ch) - 0xAC00) % 28
    cons = jong != 0 and not (pair[1] == "로" and jong == 8)
    return word + (pair[0] if cons else pair[1])


def ko_name(name, label):
    return name or KO.get(label, label)


class Follower:
    def __init__(self, idx, goal=None, label=None, area=None, off_m=2.5, detour_m=3.0, ahead_m=6.0, now_m=2.0,
                 arrive_m=1.5):
        self.idx, self.goal, self.label, self.area = idx, goal, label, area
        self.detour_m = detour_m
        self.off_m, self.ahead_m, self.now_m, self.arrive_m = off_m, ahead_m, now_m, arrive_m
        self.arrived = False
        self.reroutes = 0

    # ------------------------------------------------------------ planning
    def _target(self, dist):
        """(goal place, goal xy, name to say, symbol or None)."""
        idx = self.idx
        if self.area is not None and not self.goal and not self.label:
            aid = idx._area_arg(self.area)
            p = idx._area_target(dist, aid)
            return p, idx.place[p].center[:2], idx.s.areas[aid].get("이름") or f"구역 {aid}", None
        if self.goal:
            o = idx._o(self.goal)
        else:
            cand = [c for c in idx._match(self.label) if c.place in dist]
            if not cand:
                raise ValueError(f"갈 수 있는 {self.label}이(가) 없습니다")
            o = min(cand, key=lambda c: dist[c.place])
            self.goal = o.symbol  # keep the same object after a reroute
        name = ko_name((o.vlm or {}).get("이름"), o.label)
        return o.place, o.center[:2], f"{name}({o.symbol})", o.symbol

    def start(self, pose):
        """Plan from this pose; returns the first update (announces the first leg)."""
        idx = self.idx
        idx.pose = tuple(pose)
        src = idx._pose_place()
        dist, prev = idx._dijkstra(src)
        gp, gxy, self.name, self.sym = self._target(dist)
        if gp not in dist:
            raise ValueError("목적지에 갈 수 있는 길이 없습니다")
        path = idx._path(prev, src, gp)
        pts = np.vstack([[pose[0], pose[1]]] + [idx.place[p].center[:2] for p in path])
        if self.sym is not None:
            pts = np.vstack([pts, gxy])
        self.goal_xy = np.asarray(gxy, float)
        self.to_goal, _ = idx._dijkstra(gp)  # walking distance from every place to the goal (graph is undirected)
        self.goal_extra = float(np.linalg.norm(self.goal_xy - idx.place[gp].center[:2])) if self.sym else 0.0
        lm = [(c.symbol, c.label, (c.vlm or {}).get("이름"), c.center[:2], float(np.prod(c.size)))
              for c in idx.s.objects if c.symbol != self.sym and c.label not in SKIP_LANDMARK]
        body = pts[:-1] if self.sym is not None and len(pts) > 2 else pts
        self.steps, _ = nav.steps(body, pose[2], lm, area_at=idx._area_name_at)
        lg = nav.legs(body)
        corners = [lg[0][0]] + [b for _, b in lg]
        if self.sym is not None and np.linalg.norm(self.goal_xy - corners[-1]) > 0.3:
            corners.append(self.goal_xy)
        self.line = np.asarray(corners, float)
        seg = np.diff(self.line, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.total = float(self.cum[-1])
        # corners where a turn is said: the ends of the legs that have a next step
        self.turns = []
        for i in range(1, len(self.steps)):
            t = self.steps[i]["turn"]
            if t != "직진":
                self.turns.append({"s": float(self.cum[i]), "turn": t, "step": i, "said_ahead": False, "said_now": False})
        self.s, self.off_count, self.back_count, self.warned_back = 0.0, 0, 0, False
        self.best = float("inf")
        self.n_updates = 0
        first = self.steps[0] if self.steps else {"turn": "출발", "walk_m": nav.half_m(self.total)}
        text = self._leg_text(first, lead=True)
        if self.turns and self.turns[0]["s"] <= self.now_m + 0.5:  # the first corner is right there: one sentence
            t = self.turns[0]
            t["said_ahead"] = t["said_now"] = True
            text = text[:-1] + ", 그다음 " + self._leg_text(self.steps[t["step"]])
        return self._state(pose, [text], started=True)

    # ------------------------------------------------------------ text
    def _leg_text(self, st, lead=False):
        t = st["turn"]
        head = "" if t in ("출발", "직진") else SAY_TURN.get(t, t) + ". "
        if lead:
            head = f"{self.name}까지 약 {nav.half_m(self.total)} m입니다. " + head
        txt = head + f"약 {st['walk_m']:g} m 가세요"
        if st.get("passing"):
            p = st["passing"][0]
            txt += f". {p['side']}에 {josa(ko_name(p.get('name'), p['label']) + '(' + p['symbol'] + ')', ('이', '가'))} 보입니다"
        if st.get("into_area"):
            txt += f". {josa(st['into_area'], ('으로', '로'))} 들어갑니다"
        return txt + "."

    # ------------------------------------------------------------ progress
    def _project(self, xy):
        """(progress along the route, distance from it), searching forward from the last progress only."""
        best = None
        lo = self.s - 3.0
        for i, L in enumerate(self.seg_len):
            if self.cum[i + 1] < lo or L < 1e-9:
                continue
            a, v = self.line[i], self.line[i + 1] - self.line[i]
            t = float(np.clip(np.dot(xy - a, v) / (L * L), 0.0, 1.0))
            s = float(self.cum[i] + t * L)
            if s < lo:
                continue
            d = float(np.linalg.norm(xy - (a + t * v)))
            # prefer near the current progress when two parts of the route are equally close (ring corridor)
            key = d + 0.05 * max(0.0, s - self.s - 4.0)
            if best is None or key < best[0]:
                best = (key, s, d, i)
        if best is None:
            return self.s, float(np.linalg.norm(xy - self.line[-1])), len(self.seg_len) - 1
        return best[1], best[2], best[3]

    def _walk_left(self, pose):
        """Walking distance from this pose to the goal over the place graph (m)."""
        self.idx.pose = tuple(pose)
        p = self.idx._pose_place()
        if p not in self.to_goal:
            return float("inf")
        return float(np.linalg.norm(np.asarray(pose[:2]) - self.idx.place[p].center[:2])) + self.to_goal[p] + self.goal_extra

    def update(self, pose):
        if self.arrived:
            return self._state(pose, [])
        xy = np.asarray(pose[:2], float)
        s, d, i = self._project(xy)
        say = []
        w = self._walk_left(pose)
        self.best = min(self.best, w)
        # off the route only when also getting farther from the goal (or far longer than the plan): beside the
        # line in an open room, or reaching the goal from another side, is fine
        away = w > self.best + self.detour_m or w > max(0.0, self.total - self.s) + 2 * self.detour_m
        if d > self.off_m and away:
            self.off_count += 1
            if self.off_count >= 2:  # two updates in a row (one noisy pose is not enough)
                self.reroutes += 1
                r = self.start(pose)
                r["say"] = ["길에서 벗어났습니다. 새 길로 안내합니다. " + r["say"][0]]
                r["rerouted"] = True
                return r
        else:
            self.off_count = 0
            self.s = max(self.s, s)
        # walking away from the route direction
        v = self.line[i + 1] - self.line[i]
        rel = abs(nav.rel_angle(pose[2], v)) if np.linalg.norm(v) > 1e-6 else 0.0
        self.n_updates += 1
        self.back_count = self.back_count + 1 if rel > 135 and d <= self.off_m and self.n_updates > 4 else 0
        if self.back_count >= 3 and not self.warned_back:
            self.warned_back = True
            say.append("반대 방향으로 가고 있습니다. 뒤로 돌아 주세요.")
        if rel < 90:
            self.warned_back = False
        # arrival
        to_goal = float(np.linalg.norm(self.goal_xy - xy))
        if (self.total - self.s <= self.arrive_m and to_goal <= self.arrive_m + 1.0) or to_goal <= self.arrive_m * 0.8:
            self.arrived = True
            side = nav.direction_word(nav.rel_angle(pose[2], self.goal_xy - xy)) if to_goal > 0.5 else "바로 앞"
            say.append(f"도착했습니다. {josa(self.name, ('은', '는'))} {side}에 있습니다." if self.sym else
                       f"{self.name}에 도착했습니다.")
            return self._state(pose, say)
        # turns ahead / now
        for t in self.turns:
            gap = t["s"] - self.s
            if gap < -1.0:
                t["said_ahead"] = t["said_now"] = True
                continue
            if gap <= self.now_m and not t["said_now"]:
                t["said_now"] = t["said_ahead"] = True
                nxt = self.steps[t["step"]]
                say.append(self._leg_text(nxt))
            elif gap <= self.ahead_m and not t["said_ahead"]:
                t["said_ahead"] = True
                say.append(f"약 {nav.half_m(gap):g} m 앞에서 {SAY_TURN.get(t['turn'], t['turn'])}.")
            break  # only the next corner
        return self._state(pose, say)

    def _state(self, pose, say, started=False):
        nxt = next(({"turn": t["turn"], "in_m": round(t["s"] - self.s, 1)} for t in self.turns
                    if t["s"] - self.s > -1.0), None)
        return {"say": say, "next": nxt, "progress_m": round(self.s, 1),
                "remaining_m": round(max(0.0, self.total - self.s), 1), "arrived": self.arrived,
                "goal": self.sym or self.name, "route": [[round(float(x), 2), round(float(y), 2)] for x, y in self.line],
                **({"started": True} if started else {})}
