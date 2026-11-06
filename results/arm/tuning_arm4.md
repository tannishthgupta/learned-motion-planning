# Arm (4 joints): checkpoint and mix tuning on validation worlds

30 validation worlds that the long reference run solved, x 2 seeds, 1500 iterations. success@k = share of runs with a path after k iterations. cost@k = mean normalised cost (1.0 = best known), with a run that has no path yet counted as 3. Score = mean of the cost@k columns; lower is better.

| candidate | success@500 | success@1000 | success@1500 | cost@500 | cost@1000 | cost@1500 | score |
|---|---|---|---|---|---|---|---|
| learned e10 mix=0.50 | 73% | 83% | 90% | 1.645 | 1.437 | 1.294 | 1.459 |
| learned e10 mix=0.25 | 75% | 88% | 90% | 1.710 | 1.410 | 1.320 | 1.480 |
| learned e30 mix=0.25 | 80% | 83% | 87% | 1.640 | 1.503 | 1.393 | 1.512 |
| learned e10 mix=0.90 | 68% | 78% | 80% | 1.682 | 1.476 | 1.436 | 1.531 |
| learned e60 mix=0.25 | 75% | 83% | 87% | 1.738 | 1.531 | 1.434 | 1.568 |
| learned e30 mix=0.50 | 60% | 77% | 87% | 1.907 | 1.562 | 1.402 | 1.624 |
| RRT* | 80% | 87% | 90% | 1.953 | 1.732 | 1.521 | 1.735 |
| learned e60 mix=0.50 | 65% | 73% | 75% | 1.880 | 1.736 | 1.679 | 1.765 |
| learned e30 mix=0.90 | 57% | 65% | 70% | 1.914 | 1.745 | 1.638 | 1.766 |
| learned e60 mix=0.90 | 52% | 65% | 67% | 2.040 | 1.845 | 1.805 | 1.897 |

**Chosen: epoch 10 checkpoint, mix 0.50** (copied to learning\models\arm4_cvae.pt)
