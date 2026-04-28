"""Quick diagnostic — run on the compute node to inspect raw p_hat_z values.

Usage:
  python memorization/tests/check_raw_pz.py \
    /home/ammany/scratch/results/memorization/wiki_memorization/mdlm-wiki_random_raw.json
"""
import json, sys, math

path = sys.argv[1] if len(sys.argv) > 1 else \
  "/home/ammany/scratch/results/memorization/wiki_memorization/mdlm-wiki_random_raw.json"

with open(path) as f:
  results = json.load(f)

print(f"Total examples: {len(results)}")
print()

for r in results[:5]:
  print(f"doc={r['doc_id']}  prefix_len={r['prefix_len']}  suffix_len={r['suffix_len']}")
  for N, v in sorted(r['by_N'].items(), key=lambda x: (x[0]=='arm', int(x[0]) if x[0]!='arm' else 0)):
    lp = v.get('log_p_hat_z', float('nan'))
    pz = v.get('p_hat_z', 0.0)
    print(f"  N={N:>4}:  log_p_hat_z={lp:>10.3f}  p_hat_z={pz:.3e}")
  print()

# Also print the distribution of log_p_hat_z across all examples for N=5
N_key = '5'
log_vals = [r['by_N'][N_key]['log_p_hat_z'] for r in results if N_key in r['by_N']]
if log_vals:
  log_vals_finite = [v for v in log_vals if math.isfinite(v)]
  print(f"N={N_key} log_p_hat_z across {len(results)} examples:")
  print(f"  min={min(log_vals_finite):.2f}  max={max(log_vals_finite):.2f}  "
        f"mean={sum(log_vals_finite)/len(log_vals_finite):.2f}  "
        f"n_neginf={len(log_vals)-len(log_vals_finite)}")
