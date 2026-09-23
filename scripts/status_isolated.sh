#!/usr/bin/env bash
# Read-only status of the isolated re-run. Safe to run any time; touches nothing.
set -u
cd "$(dirname "$0")/.."
L=results/metrics
echo "=============================================="
echo " isolated re-run    $(date '+%a %H:%M')"
echo "=============================================="
running=$(pgrep -c -f 'src.fl.simulate --config' 2>/dev/null || true)
ok=$(grep -c '^OK' "$L/progress_isolated.log" 2>/dev/null | head -1); ok=${ok:-0}
fail=$(grep -c '^FAILED' "$L/progress_isolated.log" 2>/dev/null | head -1); fail=${fail:-0}
printf " arms complete: %s/5" "$ok"
[ "${fail:-0}" -gt 0 ] && printf "   (%s FAILED)" "$fail"
echo; echo
for s in 101 102 103 104 105; do
  f="$L/joblogs/isolated_s${s}_rerun.log"
  n=$(grep -c 'client [0-9]' "$f" 2>/dev/null | head -1); n=${n:-0}
  last=$(grep 'client [0-9]' "$f" 2>/dev/null | tail -1 | sed 's/.*\(client [0-9]\)/\1/')
  if grep -q "^OK isolated_s${s}$" "$L/progress_isolated.log" 2>/dev/null; then
    state="DONE"
  elif [ "$n" -gt 0 ] || [ -s "$f" ]; then state="running"
  else state="queued"; fi
  printf "  seed %s  %s/5 clients  %-8s %s\n" "$s" "$n" "$state" "$last"
done
echo
echo " processes training: ${running:-0}   load:$(uptime | sed 's/.*load average://')"
if [ "${ok:-0}" -eq 5 ]; then
  echo
  echo " ALL DONE — artifacts written:"
  echo "   client probs: $(ls $L/fl_isolated_iid_isolated_s*_client_probs.npz 2>/dev/null | wc -l)/5"
  echo "   client models: $(ls results/models/fl_isolated_iid_isolated_s*_client*.h5 2>/dev/null | wc -l)/25"
fi
