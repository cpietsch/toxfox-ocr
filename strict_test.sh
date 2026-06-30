#!/usr/bin/env bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=5 GT=v2
echo "baseline union3 (orient+rapid mobile): curated 0.8855 | scraped 0.8356"
for strat_thr in "segment:86" "segment:90" "union3:86"; do
  strat="${strat_thr%%:*}"; thr="${strat_thr##*:}"
  echo "### server-3way  strategy=$strat seg_thr=$thr ###"
  CACHE=ens ENS_ENGINES=orient,rapid,rapidserver ENS_SEGS="$thr,$thr,$thr" \
    .venv/bin/python score.py "$strat" "$thr" 2>/dev/null | grep -E "scraped|curated"
done
echo STRICTDONE
