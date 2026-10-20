"""Score-normalization boundary and reuse tests, independent of SpeechBrain."""

import pytest
import torch

from recipes.voxceleb.score_normalization import (
    CohortStatistics,
    validate_cohort_size,
)


def _cohort():
    return {
        "a": torch.tensor([[1.0, 0.0]]),
        "b": torch.tensor([[0.0, 1.0]]),
        "c": torch.tensor([[-1.0, 0.0]]),
    }


def test_cohort_size_is_capped_and_statistics_match_topk() -> None:
    normalizer = CohortStatistics(_cohort(), cohort_size=20000)
    mean, std = normalizer.get("target", torch.tensor([[1.0, 0.0]]))

    assert normalizer.effective_size == 3
    assert mean.item() == pytest.approx(0.0)
    assert std.item() == pytest.approx(1.0)


def test_cohort_statistics_are_reused_for_repeated_utterances(monkeypatch) -> None:
    normalizer = CohortStatistics(_cohort(), cohort_size=2)
    calls = 0
    original = torch.mv

    def counted_mv(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "mv", counted_mv)
    embedding = torch.tensor([[1.0, 0.0]])
    first = normalizer.get("target", embedding)
    second = normalizer.get("target", embedding)

    assert first is second
    assert calls == 1
    assert first[0].item() == pytest.approx(0.5)


def test_statistics_match_original_cosine_topk_calculation() -> None:
    cohort = _cohort()
    embedding = torch.tensor([[0.8, 0.6]])
    normalizer = CohortStatistics(cohort, cohort_size=2)

    actual_mean, actual_std = normalizer.get("target", embedding)
    stacked = torch.stack(list(cohort.values()))
    repeated = embedding.repeat(stacked.shape[0], 1, 1)
    old_scores = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)(repeated, stacked)
    old_topk = torch.topk(old_scores, k=2, dim=0).values

    torch.testing.assert_close(actual_mean, old_topk.mean())
    torch.testing.assert_close(actual_std, old_topk.std())


@pytest.mark.parametrize("size", [0, 1, -1, True, 1.5, "2"])
def test_invalid_cohort_size_is_rejected(size) -> None:
    with pytest.raises(ValueError, match="cohort_size"):
        CohortStatistics(_cohort(), size)
    with pytest.raises(ValueError, match="cohort_size"):
        validate_cohort_size(size)


def test_degenerate_or_too_small_cohort_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least two"):
        CohortStatistics({"a": torch.tensor([[1.0, 0.0]])})

    normalizer = CohortStatistics(
        {"a": torch.tensor([[1.0, 0.0]]), "b": torch.tensor([[1.0, 0.0]])}
    )
    with pytest.raises(ValueError, match="degenerate"):
        normalizer.get("target", torch.tensor([[1.0, 0.0]]))


def test_nonfinite_embeddings_are_rejected() -> None:
    bad_cohort = _cohort()
    bad_cohort["c"] = torch.tensor([[float("nan"), 0.0]])
    with pytest.raises(ValueError, match="non-finite"):
        CohortStatistics(bad_cohort)

    normalizer = CohortStatistics(_cohort())
    with pytest.raises(ValueError, match="invalid embedding"):
        normalizer.get("target", torch.tensor([[float("inf"), 0.0]]))


@pytest.mark.parametrize("mode", ["z-norm", "t-norm", "s-norm"])
def test_verification_scoring_with_normalization_modes(tmp_path, monkeypatch, mode) -> None:
    pytest.importorskip("speechbrain")
    from recipes.voxceleb import speaker_verification_cosine as recipe

    monkeypatch.setattr(
        recipe,
        "params",
        {"score_norm": mode, "cohort_size": 20000, "output_folder": str(tmp_path)},
        raising=False,
    )
    monkeypatch.setattr(recipe, "train_dict", _cohort(), raising=False)
    monkeypatch.setattr(
        recipe,
        "enrol_dict",
        {"e1": torch.tensor([[1.0, 0.0]]), "e2": torch.tensor([[0.0, 1.0]])},
        raising=False,
    )
    monkeypatch.setattr(
        recipe,
        "test_dict",
        {"t1": torch.tensor([[0.8, 0.6]]), "t2": torch.tensor([[-0.8, 0.6]])},
        raising=False,
    )

    positive, negative = recipe.get_verification_scores(
        ["1 e1 t1", "0 e2 t2", "1 e1 t1"]
    )

    assert len(positive) == 2
    assert len(negative) == 1
    assert all(torch.isfinite(value) for value in positive + negative)
    assert len((tmp_path / "scores.txt").read_text(encoding="utf-8").splitlines()) == 3
