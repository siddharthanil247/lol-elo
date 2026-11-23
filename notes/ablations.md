# Player Elo feature ablations

Scored games from 2019 onward; earlier games are burn-in. Continuity bonus was held at 50 for every run.

| configuration | region_strength | offseason_regression | continuity_bonus | games | game_log_loss | brier | accuracy | series_log_loss | international_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neither | False | False | 50 | 19930 | 0.6337 | 0.2217 | 0.6404 | 0.5980 | 0.6399 |
| regional only | True | False | 50 | 19930 | 0.6312 | 0.2206 | 0.6440 | 0.5940 | 0.6197 |
| offseason only | False | True | 50 | 19930 | 0.6349 | 0.2221 | 0.6425 | 0.5992 | 0.6483 |
| both | True | True | 50 | 19930 | 0.6320 | 0.2208 | 0.6468 | 0.5943 | 0.6218 |

## Decision

Regional strength improves all five reported metrics relative to neither feature:
game log loss, Brier score, series log loss, and international log loss decrease,
while accuracy increases. Offseason regression raises accuracy slightly, but
worsens game log loss, Brier score, series log loss, and international log loss.
The combined configuration is weaker than regional strength alone on those four
loss metrics. Regional strength remains enabled; offseason regression is disabled.
