#!/usr/bin/env bash
# Test server-OCR ensemble configurations once /tmp/cache_rapidserver_*.json exist.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=5 GT=v2
echo "baseline (orient+rapid mobile, order-fixed): curated 0.8855 | scraped 0.8356"
echo ""
run() {  # $1=label  $2=ENS_ENGINES  $3=ENS_SEGS
  echo "### $1  [$2 @ $3] ###"
  CACHE=ens ENS_ENGINES="$2" ENS_SEGS="$3" .venv/bin/python score.py union3 80 2>/dev/null | grep -E "scraped|curated"
}
run "docTR+server"          "orient,rapidserver"        "80,90"
run "docTR+mobile+server"   "orient,rapid,rapidserver"  "80,90,90"
run "server-primary+docTR"  "rapidserver,orient"        "80,90"
run "server alone"          "rapidserver"               "80"
echo "SRV_ENS_DONE"
