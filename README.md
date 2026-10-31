# Petition Moderation Assistant

Rejection-reason classification and duplicate retrieval for UK Parliament e-petitions (2015–2026).

CS5998 Capstone Project – Master of Data Science & Artificial Intelligence, University of Moratuwa
N.V.Y. Priyashan (268481U)

## Problem

About two in every three petitions submitted to the UK Parliament petitions website are rejected, and roughly half of those are duplicates of a petition that is already open. Moderators check every petition by hand. This project uses the petition text to (1) predict the outcome and rejection reason, and (2) find the open petitions a new petition duplicates. The moderator keeps the final decision. No external LLM API is used.

## Data

- Source: https://petition.parliament.uk (JSON API and CSV exports)
- Licence: Open Government Licence v3.0
- Coverage: 2015–17, 2017–19, 2019–24 and the current Parliament (128,077 petitions, snapshot 25 Sep 2026)
- The data is not stored in git. Run `src/collect.py` to download it. The results use the snapshot of 25 Sep 2026.
  The site keeps changing, so a new download will differ slightly; `02_cleaning` checks the snapshot's SHA-256 hash.

## How to run

Python 3.12 (tested with 3.12.10 on Windows 11). Create a virtual environment in the repo folder and install the
pinned packages:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS or Linux, use `.venv/bin/python` instead of `.venv\Scripts\python.exe`.

Data collection (Milestone 1):

```
.venv\Scripts\python.exe src\collect.py      # about 1 hour, writes data/petitions.jsonl
.venv\Scripts\python.exe src\check_data.py   # prints petition counts and rejection reasons
```

Then run the notebooks in order, with the `.venv` Python as the kernel (in VS Code: Select Kernel → `.venv`):

| Notebook | What it does |
|---|---|
| `notebooks/02_cleaning.ipynb` | 7-class labels, gold duplicate pairs, text cleaning, leakage test |
| `notebooks/04_split.ipynb` | train / validation / test split by decision time, time-aware candidate pools |

## Status

- [x] Milestone 1 – Project definition and data collection
- [ ] Milestone 2 – Preprocessing, EDA, baseline and improved models
- [ ] Milestone 3 – Final evaluation, dashboard and report