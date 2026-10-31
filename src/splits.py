"""Time-based splits and time-aware candidate pools (locked rules 3 and 5).

Used by notebooks/04_split.ipynb, the retrieval and fusion notebooks and later by
src/pipeline.py, so every step uses exactly the same split dates and pool rule.
"""
import numpy as np
import pandas as pd

# Rule 3: split by decision time (UTC), never at random.
TRAIN_END = pd.Timestamp("2025-07-01", tz="UTC")   # train: decided before 1 Jul 2025
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")     # validation: 1 Jul - 31 Dec 2025
TEST_END = pd.Timestamp("2026-09-01", tz="UTC")    # test: 1 Jan - 31 Aug 2026; later decisions are excluded
SPLITS = ["train", "val", "test", "excluded"]

# Rule 5: a moderator may link a petition that is published up to 48 h after the decision.
GRACE = np.timedelta64(48, "h")


def assign_split(decision_time):
    """train / val / test / excluded for every decision time in a pandas Series."""
    split = np.select(
        [decision_time < TRAIN_END, decision_time < VAL_END, decision_time < TEST_END],
        ["train", "val", "test"],
        default="excluded",
    )
    return pd.Series(split, index=decision_time.index, name="split")


def to_numpy_utc(times):
    """UTC pandas times (a Series or one Timestamp) -> plain numpy datetime64, which is fast to compare."""
    if isinstance(times, pd.Series):
        return times.dt.tz_convert(None).to_numpy()
    return times.tz_convert(None).to_datetime64()


class CandidatePools:
    """The published petitions that a query decided at time t can be a duplicate of.

    Open pool (the main one): opened by t + 48 h and not closed before t.
    Stress pool: opened by t + 48 h, open or closed.
    The query itself is never in its own pool.
    """

    def __init__(self, petitions):
        # Members: label "published" (state open or closed). Not "opened_at is set": 39 petitions were
        # published and later rejected, and most have no closed_at, so they would sit in every later pool.
        published = petitions[petitions["label"] == "published"]
        self.ids = published["id"].to_numpy()                 # this order = the order of every pool mask
        self.opened = to_numpy_utc(published["opened_at"])
        self.closed = to_numpy_utc(published["closed_at"])    # NaT = still open
        assert not np.isnat(self.opened).any()                 # every published petition has an opening time

    def mask(self, t, query_id=None, stress=False):
        """True for each published petition (in self.ids order) in the pool of a query decided at t.

        t is a UTC pandas Timestamp, e.g. a decision_time value. Use query_id=None for a new petition.
        """
        t = to_numpy_utc(t)
        in_pool = self.opened <= t + GRACE
        if not stress:
            in_pool &= np.isnat(self.closed) | (self.closed >= t)
        return in_pool & (self.ids != query_id)

    def pool_ids(self, t, query_id=None, stress=False):
        """IDs of the petitions in the pool."""
        return self.ids[self.mask(t, query_id, stress)]

    def sizes(self, times, query_ids, stress=False):
        """Pool size for each (decision time, query id) pair."""
        return np.array([self.mask(t, q, stress).sum() for t, q in zip(times, query_ids)])
