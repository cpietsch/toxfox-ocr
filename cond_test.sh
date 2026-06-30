#!/usr/bin/env bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=5 GT=v2
for n in 8 10 12 14; do
  echo "### CONDITIONAL orient,rapid + server-if-primary<$n (server@90) ###"
  CACHE=ens ENS_ENGINES=orient,rapid,rapidserver ENS_SEGS=80,90,90 \
    ENS_COND_N="$n" ENS_COND_PRIMARY=2 .venv/bin/python score.py union3 80 2>/dev/null \
    | grep -E "scraped|curated"
done
echo CONDDONE
