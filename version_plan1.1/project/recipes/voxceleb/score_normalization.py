"""Bounded, reusable cohort statistics for verification score normalization."""

from collections.abc import Mapping

import torch
import torch.nn.functional as F


def validate_cohort_size(cohort_size: int | None) -> None:
    if cohort_size is not None and (
        isinstance(cohort_size, bool)
        or not isinstance(cohort_size, int)
        or cohort_size < 2
    ):
        raise ValueError("cohort_size must be an integer of at least 2")


class CohortStatistics:
    """Compute each utterance's cohort statistics at most once."""

    def __init__(
        self,
        embeddings: Mapping[str, torch.Tensor],
        cohort_size: int | None = None,
    ) -> None:
        if len(embeddings) < 2:
            raise ValueError("score normalization requires at least two cohort embeddings")
        validate_cohort_size(cohort_size)

        cohort = torch.stack(list(embeddings.values())).reshape(len(embeddings), -1)
        if not torch.isfinite(cohort).all():
            raise ValueError("score normalization cohort contains non-finite embeddings")
        self.cohort = F.normalize(cohort, dim=-1, eps=1e-6)
        self.effective_size = min(cohort_size or len(embeddings), len(embeddings))
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def get(
        self, utterance_id: str, embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if utterance_id not in self._cache:
            vector = embedding.reshape(-1)
            if vector.numel() != self.cohort.shape[1] or not torch.isfinite(vector).all():
                raise ValueError(
                    f"invalid embedding for score normalization: {utterance_id}"
                )
            vector = F.normalize(vector, dim=0, eps=1e-6)
            # A matrix-vector product avoids materializing a repeated embedding
            # for every cohort entry and every verification pair.
            scores = torch.mv(self.cohort, vector)
            if self.effective_size < scores.numel():
                scores = torch.topk(scores, k=self.effective_size).values
            mean = scores.mean()
            std = scores.std()
            if not torch.isfinite(mean) or not torch.isfinite(std) or std <= 1e-6:
                raise ValueError(
                    f"degenerate score normalization cohort for {utterance_id}"
                )
            self._cache[utterance_id] = (mean, std)
        return self._cache[utterance_id]
