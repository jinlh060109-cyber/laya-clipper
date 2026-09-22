from clipper.classify import (FALLBACK, PROFILE_ORDER, QUESTION_ID, choose_profile,
                              sample_windows)


def _windows(n):
    return [{"id": i, "start": i * 10.0, "end": i * 10.0 + 30.0,
             "text": f"SPEAKER_00: line {i}", "preceding": "", "position": i / max(n, 1),
             "energy_mean": 0.3, "energy_peak": 0.5, "energy_peak_offset": 0.5}
            for i in range(n)]


class VotingAgent:
    """Answers the content-type question from a scripted list, cycling."""

    def __init__(self, picks):
        self.picks = list(picks)
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        pick = self.picks[(len(self.calls) - 1) % len(self.picks)]
        if isinstance(pick, Exception):
            raise pick
        return {"answers": {QUESTION_ID: {"type": "choice", "choice": pick,
                                          "confidence": 0.5}}}


def test_sample_returns_everything_when_there_are_fewer_than_n():
    assert sample_windows(_windows(5), 12) == _windows(5)


def test_sample_is_bounded_and_spread_across_the_source():
    """A cold open must not decide the type of the whole video."""
    ids = [w["id"] for w in sample_windows(_windows(100), 12)]
    assert len(ids) == 12
    assert ids == sorted(set(ids))
    assert ids[0] < 10 and ids[-1] > 90


def test_sample_of_nothing_is_empty():
    assert sample_windows([], 12) == []


def test_the_majority_wins():
    out = choose_profile(_windows(3), VotingAgent(["stream", "stream", "podcast"]))
    assert out["profile"] == "stream"
    assert out["votes"]["stream"] == 2
    assert out["sampled"] == 3
    assert out["fallback"] is False


def test_a_tie_goes_to_the_first_profile_in_the_fixed_order():
    out = choose_profile(_windows(2), VotingAgent(["stream", "lecture"]))
    assert out["profile"] == "lecture"


def test_failed_and_unknown_answers_cast_no_vote():
    agent = VotingAgent([RuntimeError("boom"), "filler", "talking_head"])
    out = choose_profile(_windows(3), agent)
    assert out["profile"] == "talking_head"
    assert sum(out["votes"].values()) == 1


def test_no_votes_at_all_falls_back_and_says_so():
    out = choose_profile(_windows(4), VotingAgent([RuntimeError("x")]))
    assert out["profile"] == FALLBACK
    assert out["fallback"] is True


def test_the_question_offers_every_profile_in_order_over_the_scoring_state():
    agent = VotingAgent(["podcast"])
    choose_profile(_windows(1), agent)
    state, questions = agent.calls[0]
    question = questions[QUESTION_ID]
    assert question["type"] == "choice"
    assert list(question["criteria"]) == list(PROFILE_ORDER)
    assert state["window"]["text"] == "SPEAKER_00: line 0"


def test_every_offered_profile_exists_on_disk():
    from clipper.profiles.loader import available_profiles
    assert set(PROFILE_ORDER) <= set(available_profiles())
