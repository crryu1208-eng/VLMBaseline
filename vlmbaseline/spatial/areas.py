"""Split the floor into areas of a walkable size (the export has one room per scene).

Seeds by geodesic farthest-point sampling over the place graph (no two seeds closer than `radius` m of walking),
every place joins its nearest seed by walking distance (geodesic Voronoi, so each area is connected), and areas
with fewer than `min_places` places join the neighbour they share most edges with.

    areas = split_areas(idx)            # {area_id: [place ids]}, area ids 1..N ordered along x then y
"""
from __future__ import annotations

import heapq
from collections import Counter, defaultdict

import numpy as np


def _multi_dijkstra(adj, seeds):
    dist, owner, pq = {}, {}, []
    for k, s in enumerate(seeds):
        dist[s], owner[s] = 0.0, k
        pq.append((0.0, s))
    heapq.heapify(pq)
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if d + w < dist.get(v, 1e18):
                dist[v], owner[v] = d + w, owner[u]
                heapq.heappush(pq, (d + w, v))
    return dist, owner


def split_areas(idx, radius=7.0, min_places=12):
    adj, place = idx.adj, idx.place
    nodes = list(adj)
    if not nodes:
        return {}
    # start from the place farthest (walking) from the graph's first place
    d0, _ = _multi_dijkstra(adj, [nodes[0]])
    seeds = [max(d0, key=d0.get)]
    while True:
        dist, _ = _multi_dijkstra(adj, seeds)
        far = max(dist, key=dist.get)
        if dist[far] < radius:
            break
        seeds.append(far)
    _, owner = _multi_dijkstra(adj, seeds)
    groups = defaultdict(list)
    for p, k in owner.items():
        groups[k].append(p)
    # merge small areas into the neighbour with the most shared edges
    while True:
        small = [k for k, ps in groups.items() if len(ps) < min_places]
        if not small or len(groups) == 1:
            break
        k = min(small, key=lambda g: len(groups[g]))
        share = Counter(owner[v] for p in groups[k] for v, _ in adj[p] if owner[v] != k)
        if not share:
            break
        t = share.most_common(1)[0][0]
        for p in groups.pop(k):
            owner[p] = t
            groups[t].append(p)
    order = sorted(groups, key=lambda g: tuple(np.round(np.mean([place[p].center[:2] for p in groups[g]], 0), 1)))
    return {i + 1: sorted(groups[g]) for i, g in enumerate(order)}
