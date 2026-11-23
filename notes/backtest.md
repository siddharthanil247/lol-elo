# Roster projection backtest

For each season 2020–2026, ratings and rookie-prior fits use only games and debut outcomes dated before that season's first game. Rosters are the most common starters across each team's first three regular-season games. Win rate uses balanced-side game probabilities.

| cohort | league | method | teams | spearman | mae |
| --- | --- | --- | --- | --- | --- |
| all teams | overall | player projection | 486 | 0.6071 | 0.1121 |
| all teams | overall | last season | 486 | 0.4759 | 0.1335 |
| all teams | overall | team Elo | 486 | 0.4990 | 0.1375 |
| all teams | CBLOL | player projection | 61 | 0.3882 | 0.1249 |
| all teams | CBLOL | last season | 61 | 0.2911 | 0.1531 |
| all teams | CBLOL | team Elo | 61 | 0.3014 | 0.1491 |
| all teams | LCK | player projection | 73 | 0.7744 | 0.0833 |
| all teams | LCK | last season | 73 | 0.5925 | 0.1071 |
| all teams | LCK | team Elo | 73 | 0.6040 | 0.1240 |
| all teams | LCP | player projection | 22 | 0.4539 | 0.1691 |
| all teams | LCP | last season | 22 | 0.5243 | 0.1610 |
| all teams | LCP | team Elo | 22 | 0.5199 | 0.1623 |
| all teams | LCS | player projection | 57 | 0.4331 | 0.1178 |
| all teams | LCS | last season | 57 | 0.5175 | 0.1310 |
| all teams | LCS | team Elo | 57 | 0.4328 | 0.1413 |
| all teams | LEC | player projection | 73 | 0.6923 | 0.1004 |
| all teams | LEC | last season | 73 | 0.4479 | 0.1324 |
| all teams | LEC | team Elo | 73 | 0.5226 | 0.1219 |
| all teams | LPL | player projection | 116 | 0.6788 | 0.0811 |
| all teams | LPL | last season | 116 | 0.5480 | 0.1083 |
| all teams | LPL | team Elo | 116 | 0.5277 | 0.1255 |
| all teams | LTA N | player projection | 13 | 0.6905 | 0.1411 |
| all teams | LTA N | last season | 13 | -0.3863 | 0.1993 |
| all teams | LTA N | team Elo | 13 | 0.5266 | 0.1529 |
| all teams | LTA S | player projection | 14 | 0.5780 | 0.1193 |
| all teams | LTA S | last season | 14 | -0.4472 | 0.1625 |
| all teams | LTA S | team Elo | 14 | 0.3683 | 0.1518 |
| all teams | PCS | player projection | 57 | 0.5616 | 0.1775 |
| all teams | PCS | last season | 57 | 0.6245 | 0.1685 |
| all teams | PCS | team Elo | 57 | 0.5908 | 0.1665 |
| 3+ roster changes | overall | player projection | 309 | 0.4438 | 0.1223 |
| 3+ roster changes | overall | last season | 309 | 0.2568 | 0.1494 |
| 3+ roster changes | overall | team Elo | 309 | 0.2726 | 0.1491 |
| 3+ roster changes | CBLOL | player projection | 46 | 0.2071 | 0.1381 |
| 3+ roster changes | CBLOL | last season | 46 | 0.0858 | 0.1712 |
| 3+ roster changes | CBLOL | team Elo | 46 | 0.0839 | 0.1636 |
| 3+ roster changes | LCK | player projection | 45 | 0.6591 | 0.0891 |
| 3+ roster changes | LCK | last season | 45 | 0.2343 | 0.1342 |
| 3+ roster changes | LCK | team Elo | 45 | 0.2889 | 0.1498 |
| 3+ roster changes | LCP | player projection | 17 | 0.0871 | 0.1711 |
| 3+ roster changes | LCP | last season | 17 | 0.1973 | 0.1735 |
| 3+ roster changes | LCP | team Elo | 17 | 0.2474 | 0.1708 |
| 3+ roster changes | LCS | player projection | 40 | 0.4810 | 0.1138 |
| 3+ roster changes | LCS | last season | 40 | 0.4873 | 0.1244 |
| 3+ roster changes | LCS | team Elo | 40 | 0.4192 | 0.1444 |
| 3+ roster changes | LEC | player projection | 41 | 0.5067 | 0.1113 |
| 3+ roster changes | LEC | last season | 41 | 0.0858 | 0.1557 |
| 3+ roster changes | LEC | team Elo | 41 | 0.1875 | 0.1263 |
| 3+ roster changes | LPL | player projection | 63 | 0.5551 | 0.0811 |
| 3+ roster changes | LPL | last season | 63 | 0.2938 | 0.1196 |
| 3+ roster changes | LPL | team Elo | 63 | 0.3188 | 0.1296 |
| 3+ roster changes | LTA N | player projection | 8 | 0.2994 | 0.1306 |
| 3+ roster changes | LTA N | last season | 8 | nan | 0.1764 |
| 3+ roster changes | LTA N | team Elo | 8 | 0.3019 | 0.1371 |
| 3+ roster changes | LTA S | player projection | 9 | 0.2667 | 0.1353 |
| 3+ roster changes | LTA S | last season | 9 | nan | 0.1789 |
| 3+ roster changes | LTA S | team Elo | 9 | 0.1951 | 0.1482 |
| 3+ roster changes | PCS | player projection | 40 | 0.3577 | 0.2011 |
| 3+ roster changes | PCS | last season | 40 | 0.5053 | 0.1851 |
| 3+ roster changes | PCS | team Elo | 40 | 0.4152 | 0.1837 |

Roster changes count replaced player identities, regardless of role. The
corrected cohort is 309/486 team-seasons with 3+ changes, down from 314. A
15-case sample (seed 2027) included first-three-game substitutes; where a
starter appeared in two of three games, the selected lineup used that majority.

G2 Esports in 2020 changed Caps and Perkz between mid and bot while keeping the
same five identities. The former role-slot count marked two changes; the new
player-set count reports zero. In the first three PCS Impunity games of 2023,
jungler Alex had a one-game source ID followed by a consistent ID in the next
two games. The one-game ID is now mapped to the established key in
`data/processed/player_aliases.csv`; the selected lineup uses the 2-of-3 key.
