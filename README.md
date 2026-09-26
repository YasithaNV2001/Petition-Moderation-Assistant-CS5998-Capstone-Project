# Petition Moderation Assistant

Rejection-reason classification and duplicate retrieval for UK Parliament e-petitions (2015–2026).

CS5998 Capstone Project – MSc in Data Science & Artificial Intelligence, University of Moratuwa
N.V.Y. Priyashan (268481U)

## Problem

About two in every three petitions submitted to the UK Parliament petitions website are rejected, and roughly half of those are duplicates of a petition that is already open. Moderators check every petition by hand. This project uses the petition text to (1) predict the outcome and rejection reason, and (2) find the open petitions a new petition duplicates. The moderator keeps the final decision. No external LLM API is used.

## Data

- Source: https://petition.parliament.uk (JSON API and CSV exports)
- Licence: Open Government Licence v3.0
- Coverage: 2015–17, 2017–19, 2019–24 and the current Parliament (128,077 petitions, snapshot 25 Sep 2026)
- The data is not stored in git. Run `src/collect.py` to download it.

## How to run

Python 3.11, standard library only (for now).

```
python src/collect.py      # about 1 hour, writes data/petitions.jsonl
python src/check_data.py   # prints petition counts and rejection reasons
```

## Status

- [x] Milestone 1 – Project definition and data collection
- [ ] Milestone 2 – Preprocessing, EDA, baseline and improved models
- [ ] Milestone 3 – Final evaluation, dashboard and report