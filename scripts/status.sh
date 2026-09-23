#!/usr/bin/env bash
# Live status of the run_all_arms_parallel.sh run.
#   bash scripts/status.sh            one shot
#   watch -n 30 bash scripts/status.sh
set -u
cd "$(dirname "$0")/.."
L=results/metrics
TOTAL=$(wc -l < "$L/joblist.txt" 2>/dev/null | tr -d ' ' || echo 0)

hms () { printf "%dh%02dm" $(( $1/3600 )) $(( ($1%3600)/60 )); }
# grep -c prints 0 AND exits 1 on no match, so "|| echo 0" would double it
count () { local n; n=$(grep -c "$1" "$2" 2>/dev/null | head -1); echo "${n:-0}"; }

now=$(date +%s)
start=$(stat -c %Y "$L/joblist.txt" 2>/dev/null || echo "$now")
elapsed=$(( now - start ))

ok=$(count "^OK" "$L/progress.log")
fail=$(count "^FAILED" "$L/progress.log")
done_n=$(( ok + fail ))

echo "=================================================================="
echo " 5G_ML run   started $(date -d @"$start" +%H:%M)   elapsed $(hms $elapsed)"
echo "=================================================================="
printf " phase A: %d/%d arms complete" "$done_n" "$TOTAL"
[ "$fail" -gt 0 ] && printf "   (%d FAILED)" "$fail"
echo; echo

echo " RUNNING"
found=0
while read -r pid args; do
  case "$args" in
    *--tag*)
      tag=$(echo "$args" | sed -n 's/.*--tag \([^ ]*\).*/\1/p')
      line=$(grep 'round' "$L/joblogs/$tag.log" 2>/dev/null | tail -1)
      r=$(echo "$line" | sed -n 's|.*round \([0-9]*/[0-9]*\).*|\1|p')
      auc=$(echo "$line" | sed -n 's/.*auc=\([0-9.]*\).*/\1/p')
      printf "   %-24s round %-7s %s\n" "$tag" "${r:-starting}" "${auc:+auc=$auc}"
      found=$(( found + 1 )) ;;
    *--arm\ centralized*)
      s=$(count "seed=" "$L/repeated_centralized.log")
      printf "   %-24s seed %d/5 done, seed %d training
" "centralized_e20" "$s" "$(( s + 1 ))"
      found=$(( found + 1 )) ;;
    *--arm\ federated*)
      s=$(count "seed=" "$L/repeated_federated_r20.log")
      line=$(grep 'round' "$L/repeated_federated_r20.log" 2>/dev/null | tail -1)
      r=$(echo "$line" | sed -n 's|.*round \([0-9]*/[0-9]*\).*|\1|p')
      printf "   %-24s seed %d/5 done, now seed %d round %s
" "federated_iid_r20" "$s" "$(( s + 1 ))" "${r:-starting}"
      found=$(( found + 1 )) ;;
    *scalability*)
      k=$(echo "$args" | sed -n 's/.*--clients \([0-9]*\).*/\1/p')
      sd=$(echo "$args" | sed -n 's/.*--seed \([0-9]*\).*/\1/p')
      printf "   %-24s K=%s seed=%s\n" "scalability" "$k" "$sd"
      found=$(( found + 1 )) ;;
  esac
done < <(pgrep -af 'src.fl.simulate|repeated_runs.py|scripts.scalability' \
         | grep -v 'bash -c' | grep -v pgrep)
[ "$found" -eq 0 ] && echo "   (nothing running)"
echo

if [ "$done_n" -gt 0 ]; then
  echo " LAST COMPLETED"
  tail -5 "$L/progress.log" | sed 's/^/   /'
  echo
fi

echo " QUEUE: $(( TOTAL - done_n - found )) arms waiting"
if [ "$done_n" -ge 2 ] && [ "$elapsed" -gt 0 ]; then
  rate=$(( elapsed / done_n ))
  left=$(( (TOTAL - done_n) * rate ))
  echo " ETA:   ~$(hms $left) remaining  (finish ~$(date -d "@$(( now + left ))" '+%a %H:%M'))"
  echo "        + ~2-3h for the scalability sweep afterwards"
else
  echo " ETA:   ~26h total from start (too few arms finished to measure yet)"
fi
echo
echo " load:$(uptime | sed 's/.*load average: //')   mem: $(free -h | awk '/^Mem:/{print $7" avail of "$2}')"
