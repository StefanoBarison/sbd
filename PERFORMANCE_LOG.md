# Performance log

Newest results at bottom. Format:
`YYYY-MM-DD HH:MM:SS | commit | step | total_time | metrics`

2026-07-27 20:49:16 | ef506f0 | single-spin thick restart (--restart_keep) local N2 top1000 singlet K=3906 method1 OMP=4 | matvec=0.33s | matvec_calls: 311(keep=1)->298(keep=15 default); energy -108.792795252 unchanged; sweet spot keep~block/2

2026-07-28 15:00:00 | (pending) | single-spin thick restart auto-size nb from keep, N2 top2000 singlet K=8907 method1 OMP=4 | matvec_calls: keep=1->181, keep=15->194, keep=22->161 | energy -108.8084795799922 identical (rel 1e-15) for keep=1/15/22; auto nb=30/44/51 keeps growth=29 constant -> keep>1 no longer stalls (was garbage before)
