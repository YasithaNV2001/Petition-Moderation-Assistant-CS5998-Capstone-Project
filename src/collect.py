"""Download every UK Parliament petition (2015-17, 2017-19, 2019-24 and the current Parliament).

The API listing pages are sorted only by signature count, with no tie-breaker,
so paging through them skips some petitions and repeats others. To get a
complete dataset this script:
  1. takes the full list of petition IDs from each Parliament's CSV export,
  2. pages through the JSON listings to collect most petitions quickly,
  3. fetches every petition that is still missing one by one,
  4. saves everything to data/petitions.jsonl (one petition per line).
"""
import csv, glob, io, json, os, re, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://petition.parliament.uk/"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
DATA = os.path.join(ROOT, "data")
PAGES = os.path.join(DATA, "pages")    # cache of listing pages
SINGLE = os.path.join(DATA, "single")  # cache of individually fetched petitions
os.makedirs(PAGES, exist_ok=True)
os.makedirs(SINGLE, exist_ok=True)
HEADERS = {"User-Agent": "CS5998 capstone research"}
ARCHIVED = {"2015-17": 1, "2017-19": 3, "2019-24": 4}  # API parliament numbers


def get(url, as_json=True):
    """Download a URL, retrying on errors. Returns None if the page does not exist."""
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            body = urllib.request.urlopen(req, timeout=60).read()
            return json.loads(body) if as_json else body.decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {url}")


def ids_from_csv(path, states=None):
    """Petition IDs from a CSV export (the URL column ends with the ID)."""
    rows = csv.DictReader(io.StringIO(get(BASE + path, as_json=False)))
    return [int(r["URL"].rstrip("/").split("/")[-1]) for r in rows if states is None or r["State"] in states]


# 1. Complete list of petition IDs
expected = {}
for name, num in ARCHIVED.items():
    for pid in ids_from_csv(f"archived/petitions.csv?parliament={num}&state=all"):
        expected[pid] = name
for pid in ids_from_csv("petitions.csv?state=all", states={"open", "closed", "rejected"}):
    expected.setdefault(pid, "current")
print("Petitions listed in the CSV exports:", len(expected))

# 2. Listing pages (25 petitions each)
listings = {name: f"archived/petitions.json?parliament={num}&state=all" for name, num in ARCHIVED.items()}
listings.update({f"current-{s}": f"petitions.json?state={s}" for s in ("open", "closed", "rejected")})


def download_page(job):
    name, path, page = job
    fn = os.path.join(PAGES, f"{name}_{page:05d}.json")
    if not os.path.exists(fn):
        d = get(f"{BASE}{path}&page={page}")  # download first, so a failed download leaves no empty file
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(d, f)
        time.sleep(0.1)


jobs = []
for name, path in listings.items():
    last_link = get(BASE + path)["links"]["last"] or "page=1"
    last = int(re.search(r"page=(\d+)", last_link).group(1))
    jobs += [(name, path, page) for page in range(1, last + 1)]
with ThreadPoolExecutor(6) as pool:
    for i, _ in enumerate(pool.map(download_page, jobs), 1):
        if i % 500 == 0:
            print(f"  listing pages {i}/{len(jobs)}")

records = {}
for fn in glob.glob(os.path.join(PAGES, "*.json")):
    for p in json.load(open(fn, encoding="utf-8"))["data"]:
        if p["id"] in expected:
            records[p["id"]] = p
missing = [pid for pid in expected if pid not in records]
print(f"From listing pages: {len(records)} | still missing: {len(missing)}")


# 3. Fetch the missing petitions one by one
def download_single(pid):
    fn = os.path.join(SINGLE, f"{pid}.json")
    if not os.path.exists(fn):
        prefix = "petitions" if expected[pid] == "current" else "archived/petitions"
        d = get(f"{BASE}{prefix}/{pid}.json")
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(d["data"] if d else None, f)
        time.sleep(0.1)


with ThreadPoolExecutor(6) as pool:
    for i, _ in enumerate(pool.map(download_single, missing), 1):
        if i % 2000 == 0:
            print(f"  single petitions {i}/{len(missing)}")
not_found = 0
for pid in missing:
    d = json.load(open(os.path.join(SINGLE, f"{pid}.json"), encoding="utf-8"))
    if d:
        records[pid] = d
    else:
        not_found += 1

# 4. Save one petition per line
with open(os.path.join(DATA, "petitions.jsonl"), "w", encoding="utf-8") as f:
    for pid, p in records.items():
        f.write(json.dumps({"source": expected[pid], **p}) + "\n")
print(f"Saved {len(records)} petitions | not retrievable: {not_found} | {time.strftime('%Y-%m-%d %H:%M')}")
