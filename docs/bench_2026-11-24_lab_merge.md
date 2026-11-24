# lab_03 map + the first real walk (2026-11-24)

`build_map.sh` with the lab_03 bag and the walk bag of 2026-11-24 (outputs/live/maps/lab_03w.db): 1187 nodes, 2 maps,
211 loop closures; 1048 nodes exported, 370 matched to export keyframes (live/align.py). Replays as live:

| walk | lab_03 map | lab_03 + walk map |
|---|---|---|
| lab_03 (error vs export keyframes, median / 90% / max) | 0.03 / 0.16 / 0.89 m | 0.03 / 0.40 / 0.74 m |
| lab_02 (off the floor plan, median / 90% / max; jumps > 1 m) | 0.29 / 0.48 / 0.77 m; 0 | 0.28 / 0.82 / 1.17 m; 3 |
| camera walk 2026-11-22 | 0.27 / 0.49 / 0.56 m | 0.29 / 0.47 / 0.57 m |
| walk 2026-11-24 (in the map: same images) | first match 65 s | first match 27 s |

Slow first match on 2026-11-24: walking back from the first match, the walk started about 7 m outside the mapped
floor (no walkable place there in the export), stood 24 s, walked 10 m into the office and matched 14 s after entering
the mapped part. The merged map covers that start (first match 27 s), but the walk's own part of the map is only as
good as its odometry, lost three times past the round corridor and not loop-closed there; the other walks get a little
worse in it. Kept: lab_03 map. Better: a mapping walk of the round corridor part in both directions, starting and ending
in the mapped part (loop closures at both ends), checked with another walk.
