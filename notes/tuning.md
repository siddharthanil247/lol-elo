# Player Elo random search

Random seed 42; 8 trials. The search objective uses 2019–2023 walk-forward game log loss and 2020–2023 walk-forward backtest MAE with weights 0.70/0.30. All prior fits and rating snapshots stop before each evaluation cutoff. 2024 onward is used only for the final holdout comparison.

## Trials

| trial | game_log_loss | backtest_mae | objective | scored_games | k_base | k_min_ratio | k_decay | blue_side_advantage | continuity_bonus | k_region | regress_frac | new_season_k_mult |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.6321 | 0.1181 | 0.4779 | 11835 | 122.4488 | 0.5154 | 38.7114 | 46.8437 | 91.7832 | 0.1422 | 0.1147 | 1.3174 |
| 0 | 0.6363 | 0.1116 | 0.4789 | 11835 | 160.0000 | 0.2500 | 20.0000 | 25.0000 | 50.0000 | 0.0200 | 0.0000 | 1.0000 |
| 2 | 0.6353 | 0.1172 | 0.4799 | 11835 | 132.9816 | 0.3604 | 8.3124 | 33.2751 | 12.7635 | 0.2069 | 0.1168 | 1.0973 |
| 1 | 0.6338 | 0.1213 | 0.4800 | 11835 | 118.8464 | 0.3572 | 63.4349 | 41.8421 | 18.8355 | 0.2439 | 0.1390 | 1.3219 |
| 7 | 0.6340 | 0.1248 | 0.4813 | 11835 | 72.1211 | 0.5987 | 33.9415 | 24.3832 | 162.8041 | 0.0417 | 0.0752 | 1.0762 |
| 6 | 0.6399 | 0.1135 | 0.4820 | 11835 | 71.5500 | 0.4414 | 10.6405 | 1.8491 | 87.3435 | 0.0536 | 0.0440 | 1.3310 |
| 4 | 0.6402 | 0.1173 | 0.4833 | 11835 | 93.5070 | 0.3560 | 58.3572 | 42.0159 | 62.4733 | 0.2081 | 0.0300 | 1.0037 |
| 3 | 0.6664 | 0.1234 | 0.5035 | 11835 | 58.5802 | 0.0807 | 6.5728 | 40.9829 | 148.9524 | 0.2419 | 0.0714 | 1.1135 |

## Holdout

| configuration | cohort | game_log_loss | series_log_loss | backtest_spearman | backtest_mae | holdout_objective |
| --- | --- | --- | --- | --- | --- | --- |
| current | all teams | 0.6235 | 0.5679 | 0.6459 | 0.1132 | 0.4704 |
| current | 3+ changes | 0.6235 | 0.5679 | 0.4350 | 0.1232 | 0.4704 |
| tuned | all teams | 0.6251 | 0.5739 | 0.5856 | 0.1274 | 0.4758 |
| tuned | 3+ changes | 0.6251 | 0.5739 | 0.3545 | 0.1348 | 0.4758 |

## Decision

Tuned parameters do not win the weighted holdout objective (0.47577 vs 0.47044); the default config remains unchanged. Tuned files are available for inspection.

## Selected parameters

```yaml
k_factor: 32
starting_rating: 1500
blue_side_advantage: 46.84374186131807
k_base: 122.44875482524323
k_min_ratio: 0.5153955996144225
k_decay: 38.71140079797291
role_weights:
  top: 0.08105552953213255
  jng: 0.029079137968163898
  mid: 0.24300922197598815
  bot: 0.3086275289730575
  sup: 0.33822858155065777
continuity_n: 5
continuity_bonus: 91.783155107668
region_strength_enabled: true
k_region: 0.14218529898822344
use_rookie_priors: true
offseason_regression_enabled: true
regress_frac: 0.11474982861240383
new_season_m: 10
new_season_k_mult: 1.3173591600002954
tuning_objective_weights:
  game_log_loss: 0.7
  backtest_mae: 0.3
series_formats:
  default:
    regular: 3
    playoffs: 5
  leagues:
    LEC:
      regular: 1
      playoffs: 5
    LCS:
      regular: 1
      playoffs: 5
    EU LCS:
      regular: 1
      playoffs: 5
    NA LCS:
      regular: 1
      playoffs: 5
    WLDs:
      regular: 1
      playoffs: 5
    EWC:
      regular: 1
      playoffs: 5
    MSC:
      regular: 1
      playoffs: 5
```
