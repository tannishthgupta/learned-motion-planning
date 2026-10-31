### Headline comparison (all densities pooled)

| planner | success rate | path length (common subset) | normalised length | time to first path | total time | nodes |
|---|---|---|---|---|---|---|
| A* | 100.0% | 142.18 ± 9.05 | 1.052× | 0.0323 s | 0.0323 s | 1371 |
| RRT | 100.0% | 172.00 ± 23.86 | 1.272× | 0.0030 s | 0.0030 s | 199 |
| RRT* | 100.0% | 135.08 ± 7.42 | 1.000× | 0.0043 s | 0.9986 s | 7435 |

### By obstacle density

| density | planner | success | length (own) | length (common) | norm. | first path (s) | total (s) | nodes | motion checks |
|---|---|---|---|---|---|---|---|---|---|
| sparse | A* | 100% | 134.90 | 134.90 | 1.041× | 0.0105 | 0.0105 | 788 | 6119 |
| sparse | RRT | 100% | 159.96 | 159.96 | 1.235× | 0.0020 | 0.0020 | 153 | 170 |
| sparse | RRT* | 100% | 129.51 | 129.51 | 1.000× | 0.0030 | 1.1218 | 8472 | 40343 |
| moderate | A* | 100% | 141.86 | 141.86 | 1.062× | 0.0356 | 0.0356 | 1419 | 10779 |
| moderate | RRT | 100% | 169.46 | 169.46 | 1.268× | 0.0027 | 0.0027 | 166 | 172 |
| moderate | RRT* | 100% | 133.57 | 133.57 | 1.000× | 0.0039 | 0.9992 | 7483 | 35355 |
| cluttered | A* | 100% | 149.78 | 149.78 | 1.053× | 0.0474 | 0.0474 | 1906 | 14213 |
| cluttered | RRT | 100% | 186.58 | 186.58 | 1.313× | 0.0060 | 0.0060 | 278 | 290 |
| cluttered | RRT* | 100% | 142.16 | 142.16 | 1.000× | 0.0079 | 0.8672 | 6350 | 29051 |
| all | A* | 100% | 142.18 | 142.18 | 1.052× | 0.0323 | 0.0323 | 1371 | 10370 |
| all | RRT | 100% | 172.00 | 172.00 | 1.272× | 0.0030 | 0.0030 | 199 | 211 |
| all | RRT* | 100% | 135.08 | 135.08 | 1.000× | 0.0043 | 0.9986 | 7435 | 34916 |

_Path length on the common subset is averaged over the 150 of 150 instances that all three planners solved; averaging each planner over its own successes alone would flatter whichever planner fails on the hardest worlds._
