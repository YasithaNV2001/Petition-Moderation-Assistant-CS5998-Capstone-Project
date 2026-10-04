"""Quick check of data/petitions.jsonl: petitions per Parliament and state, and rejection reasons."""
import collections, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
counts, reasons = collections.Counter(), collections.Counter()

for line in open(os.path.join(ROOT, "data", "petitions.jsonl"), encoding="utf-8"):
    p = json.loads(line)
    state = p["attributes"]["state"]
    counts[(p["source"], state)] += 1
    if state == "rejected":
        reasons[p["attributes"]["rejection"]["code"]] += 1

print("Total petitions:", sum(counts.values()))
for (source, state), n in sorted(counts.items()):
    print(f"  {source:8s} {state:9s} {n:7,d}")
print("Rejection reasons:")
for code, n in reasons.most_common():
    print(f"  {code:18s} {n:7,d}")
