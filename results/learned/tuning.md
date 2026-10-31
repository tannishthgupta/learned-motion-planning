# Mix tuning (validation set)

40 validation problems x 3 seeds, 2000 iterations. Score = mean normalised cost over checkpoints >= 250 (lower is better).

| method | cost@100 | cost@250 | cost@500 | cost@1000 | cost@2000 | score |
|---|---|---|---|---|---|---|
| RRT* | not reached | 1.275 | 1.213 | 1.129 | 1.046 | 1.1660 |
| RRT* + learned (mix=0.25) | 1.376 | 1.148 | 1.075 | 1.029 | 1.009 | 1.0654 |
| RRT* + learned (mix=0.50) | 1.203 | 1.089 | 1.041 | 1.011 | 1.003 | 1.0360 |
| RRT* + learned (mix=0.75) | 1.121 | 1.055 | 1.022 | 1.007 | 1.001 | 1.0214 |
| RRT* + learned (mix=0.90) | 1.111 | 1.046 | 1.015 | 1.005 | 1.000 | 1.0166 |

**Chosen mix: 0.90**
