"""The user's pose from RTAB-Map localization, without ROS (live/bridge.py feeds it; live/replay_raw.py replays its
raw log offline, so the same code can be tuned on recordings).

Inputs: odometry (every frame), localization poses in the saved map (when a frame matched it) and the id of the map
node it matched. map_T_odom is set by a localization; between them the pose follows odometry. The pose in the world
frame of the export uses the correction of a map node near the pose (align.py), the one whose id is closest to the
last matched node (the same pass).

Jump check (gate): in a place with look-alike parts (library shelves), a frame can match the wrong aisle. The pose
then jumps 3-5 m and stays there for several matches (7-20 s), while odometry keeps going right. So a localization
that puts the pose more than jump_m away from where odometry carries it is not taken at once: the pose keeps
following odometry and is marked uncertain (the demo server holds the guidance). It is taken when it keeps coming
back (the same place in confirm matches) after odometry alone walked trust_m since the last accepted match
(odometry drifts, so a real correction comes back like that), or at once when odometry was lost. A match that
agrees with odometry again ends the uncertainty.
Settings from 9 library replays in maps of other recordings (docs/bench_2026-11-19.md): jump_m 2.0-2.5 the same,
3.0 lets wrong aisles through; trust_m under 25 takes them too.

Lost odometry: when odometry was lost since the last accepted match (it restarts from a guess), the pose is marked
uncertain too (why "lost") until the map matches again. On the first real walk (2026-11-24, lab_03) no frame matched
for 2 min in a part the map saw only facing the other way, odometry was lost three times there, and the pose was 7 m
off (drawn outside the corridor) when the map matched again, with nothing said. Odometry alone without a loss is not
marked by default: in the 9 library replays its error stayed under 1.35 m even 30 m after the last match, and
marking it from 10 m on would hold the guidance 30% of the time (stale_m sets a distance anyway, why "far").
"""
from __future__ import annotations

import bisect
import math

import numpy as np

NEAR_M = 4.0  # the correction comes from a map node this close to the pose


class PoseCore:
    def __init__(self, align, gate=True, jump_m=2.0, confirm=3, trust_m=25.0, stale_m=None):
        self.B_S = np.array(align["body_T_sensor"])
        self.ids = np.array([n["id"] for n in align["nodes"]])
        self.pos = np.array([n["pos"] for n in align["nodes"]])
        self.corr = np.array([n["corr"] for n in align["nodes"]])
        self.gate, self.jump_m, self.confirm, self.trust_m = gate, jump_m, confirm, trust_m
        self.stale_m = stale_m  # None: never
        self.odom_t, self.odom_T, self.odom_s = [], [], []  # recent odometry by stamp, path length
        self.walked = 0.0
        self.map_T_odom, self.loc_ns, self.node = None, None, None
        self.n_loc = self.n_rejected = 0
        self.matched_once = False
        self.acc_s = 0.0  # odometry path length at the last accepted localization
        self.cand = []  # rejected localizations since: (map_T_odom, node)
        self.lost = False  # odometry was lost since the last accepted localization
        self.info, self.decided = {}, {}  # by stamp: matched node / accepted (info and pose come in either order)

    def on_odom(self, t, T, lost=False):
        if lost:
            self.lost = True
            return
        if self.odom_T:
            self.walked += float(np.linalg.norm(T[:3, 3] - self.odom_T[-1][:3, 3]))
        self.odom_t.append(t)
        self.odom_T.append(T)
        self.odom_s.append(self.walked)
        if len(self.odom_t) > 600:
            del self.odom_t[:100], self.odom_T[:100], self.odom_s[:100]

    def odom_at(self, t):
        i = bisect.bisect_left(self.odom_t, t)
        best = min((j for j in (i - 1, i) if 0 <= j < len(self.odom_t)), key=lambda j: abs(self.odom_t[j] - t),
                   default=None)
        if best is None or abs(self.odom_t[best] - t) > 100_000_000:
            return None
        return self.odom_T[best]

    def on_loc(self, t, L):
        """L: map_T_body of the localization at stamp t. Returns 'accepted' / 'rejected' / None (no odometry)."""
        O = self.odom_at(t)
        if O is None:
            return None
        mto = L @ np.linalg.inv(O)
        ok = True
        if self.gate and self.map_T_odom is not None and not self.lost:
            jump = float(np.linalg.norm((self.map_T_odom @ O)[:3, 3] - L[:3, 3]))
            if jump > self.jump_m:
                ok = False
                # the same other place again (a consistent correction, not one stray frame)?
                same = [c for c in self.cand if np.linalg.norm((c @ O)[:3, 3] - L[:3, 3]) <= self.jump_m]
                self.cand.append(mto)
                if len(same) + 1 >= self.confirm and self.walked - self.acc_s > self.trust_m:
                    ok = True
        self._keep(self.decided, t, ok)
        if not ok:
            self.n_rejected += 1
            return "rejected"
        self.map_T_odom, self.loc_ns = mto, t
        self.n_loc += 1
        self.acc_s, self.cand, self.lost = self.walked, [], False
        if t in self.info:
            self.node = self.info[t]
        return "accepted"

    def on_info(self, t, nid):
        """The map node the frame at stamp t matched (0: none). Picks the correction only for an accepted match."""
        if nid <= 0:
            return
        self.matched_once = True
        self._keep(self.info, t, nid)
        if self.decided.get(t, not self.gate):  # the pose came first and was taken (or no gate: as before)
            self.node = nid

    @staticmethod
    def _keep(d, k, v):
        d[k] = v
        if len(d) > 50:
            del d[next(iter(d))]

    def pose(self):
        """The record for the demo server, or None before any odometry."""
        if not self.odom_t:
            return None
        t = self.odom_t[-1]
        ok = self.map_T_odom is not None and self.matched_once
        rec = {"stamp_ns": t, "localized": ok, "n_loc": self.n_loc}
        if not ok:
            return rec
        M = self.map_T_odom @ self.odom_T[-1]
        p = M[:3, 3]
        d = np.linalg.norm(self.pos - p, axis=1)
        k = None
        if self.node is not None:
            near = d < NEAR_M
            if near.any():
                k = int(np.argmin(np.where(near, np.abs(self.ids - self.node), np.inf)))
        if k is None:
            k = int(np.argmin(d))
        S = self.corr[k] @ M @ self.B_S
        f = S[:3, 2]
        since_m = self.walked - self.acc_s
        conflict = len(self.cand)
        why = "jump" if conflict else "lost" if self.lost else \
            "far" if self.stale_m is not None and since_m > self.stale_m else None
        rec.update(pose=[round(float(S[0, 3]), 3), round(float(S[1, 3]), 3), round(math.atan2(f[1], f[0]), 4)],
                   since_loc_s=round((t - self.loc_ns) / 1e9, 2), since_loc_m=round(since_m, 2),
                   conflict=conflict, uncertain=why is not None, why=why,
                   node=int(self.ids[k]), matched=self.node, map_T=[round(float(v), 4) for v in M[:3].ravel()])
        return rec


def T_of(p):
    """4x4 from a pose with .position and .orientation (ROS geometry_msgs/Pose) or a dict of 7 numbers."""
    q, t = p.orientation, p.position
    w, x, y, z = q.w, q.x, q.y, q.z
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def flat(T):
    return [round(float(v), 5) for v in T[:3].ravel()]


def unflat(v):
    T = np.eye(4)
    T[:3] = np.array(v, float).reshape(3, 4)
    return T
