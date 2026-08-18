"""Consistency checks on the example problem table itself.

The table is fixture data every other test file reads, and most of it is only ever
exercised indirectly: a problem that quietly falls out of the derived lists still leaves
a green suite, because nothing then runs it. These checks are what makes such a problem
fail loudly instead.
"""

import pytest

from quadax import quadcc, quadgk, quadts, romberg, rombergts

from .problems import (
    ALL,
    DIFFICULTY_TAGS,
    KNOWN_FAILURES,
    PROBLEMS,
    TAGS,
    TOLS,
    problem_id,
)

METHODS = {m.__name__ for m in (quadgk, quadcc, quadts, romberg, rombergts)}


@pytest.mark.parametrize("i", ALL, ids=problem_id)
def test_every_problem_has_exactly_one_difficulty_tag(i):
    """The four difficulty tags answer one question, so exactly one of them applies.

    Carrying none is the failure this is really guarding against: the derived lists are
    unions over these tags, so an untagged problem is in none of them and is silently
    never run.
    """
    tags = PROBLEMS[i]["tags"] & DIFFICULTY_TAGS
    assert len(tags) == 1, f"{PROBLEMS[i]['name']} has difficulty tags {sorted(tags)}"


@pytest.mark.parametrize("i", ALL, ids=problem_id)
def test_tags_are_spelled_correctly(i):
    """Tags are plain strings, so a typo would otherwise just make an empty group."""
    unknown = PROBLEMS[i]["tags"] - TAGS
    assert not unknown, f"{PROBLEMS[i]['name']} has unknown tags {sorted(unknown)}"


def test_problem_names_are_unique():
    """Names are the pytest ids and the key `KNOWN_FAILURES` is written against."""
    names = [p["name"] for p in PROBLEMS]
    assert len(set(names)) == len(names)


def test_known_failures_point_at_real_cases():
    """A stale entry silently stops applying rather than failing.

    `xfail_if_known` looks its key up and does nothing when it misses, so renaming a
    problem turns its entries into dead weight that no longer marks anything. The
    tolerances are checked too, since only those in `TOLS` are ever swept.
    """
    names = {p["name"] for p in PROBLEMS}
    for (method, problem), tols in KNOWN_FAILURES.items():
        assert method in METHODS, f"{method} is not a routine under test"
        assert problem in names, f"{method}/{problem} names no problem in the table"
        extra = tols - set(TOLS)
        assert not extra, f"{method}/{problem} lists untested tolerances {extra}"
