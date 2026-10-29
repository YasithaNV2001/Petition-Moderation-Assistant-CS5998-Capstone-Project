"""Text cleaning, labels, duplicate links and the leakage check.

Used by notebooks/02_cleaning.ipynb and later by src/pipeline.py, so that a new
petition is cleaned in exactly the same way as the training data.
"""
import html
import re

# The 7 classes: published + 6 rejection reasons.
# fake-name, operational and foi are dropped (886 petitions): fake-name depends on
# the creator's name, which the model must not see; the other two are too rare.
KEPT_CODES = ["duplicate", "irrelevant", "no-action", "honours", "already-happening", "local"]
CLASSES = ["published"] + KEPT_CODES

# The only columns kept in the cleaned table. Every other raw field
# (signature_count, departments, committee_note, government_response, debate,
# rejection details, creator_name, ...) is set after the moderation decision or is
# personal data, so it is never copied into the table.
TEXT_COLUMNS = ["action", "background", "additional_details", "text"]
TIME_COLUMNS = ["created_at", "opened_at", "rejected_at", "closed_at", "decision_time"]
CLEAN_COLUMNS = ["id", "source", "label"] + TEXT_COLUMNS + TIME_COLUMNS + ["likely_non_english"]

# The only column a classifier may receive as input.
MODEL_INPUTS = ["text"]

ANGLE_RE = re.compile(r"<([a-zA-Z/][^<>]*)>")              # <Daily Mail>, <http://...>
URL_RE = re.compile(r"(?:https?://|www\.)\S+")
INVISIBLE_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f​﻿]")  # control chars, zero-width space, BOM
LINK_RE = re.compile(r"petitions/+(\d+)")                  # also matches the broken "petitions//123"


def clean_text(s):
    """Clean one text field. Casing is kept (the transformer needs it)."""
    if not isinstance(s, str):         # empty field: None, or NaN in pandas 3 (NaN counts as True, so "s or ''" fails)
        s = ""
    s = html.unescape(s)               # "&amp;" -> "&"
    s = ANGLE_RE.sub(r" \1 ", s)       # "<Daily Mail>" -> "Daily Mail": drop the brackets, keep the words
    s = URL_RE.sub("[URL]", s)         # links carry no meaning for the models
    s = INVISIBLE_RE.sub(" ", s)
    return " ".join(s.split())         # one space between words (also handles tabs, newlines, non-breaking spaces)


def build_text(action, background, additional_details):
    """Model input: the three fields joined with " . " (empty fields are skipped)."""
    parts = [clean_text(x) for x in (action, background, additional_details)]
    return " . ".join(p for p in parts if p)


def make_label(state, rejection_code):
    """open/closed -> "published"; rejected -> its rejection code; None if the code is dropped."""
    if state in ("open", "closed"):
        return "published"
    if rejection_code in KEPT_CODES:
        return rejection_code
    return None


def linked_ids(details):
    """Petition IDs linked in a rejection message, without repeats."""
    if not isinstance(details, str):   # no message: None, or NaN in pandas 3
        return []
    return sorted({int(x) for x in LINK_RE.findall(details)})


def leakage_check(X, allowed=MODEL_INPUTS):
    """Stop with an error if X (a DataFrame or Series) holds any column that is not allowed."""
    columns = list(X.columns) if hasattr(X, "columns") else [X.name]
    not_allowed = [c for c in columns if c not in allowed]
    if not_allowed:
        raise ValueError(f"Leakage check failed, these columns are not allowed: {not_allowed}")
