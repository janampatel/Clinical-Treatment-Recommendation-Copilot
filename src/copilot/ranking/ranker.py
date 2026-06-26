from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from ..candidates.generator import Candidate
from ..config import RANKER_MODEL_PATH
from .features import FEATURE_NAMES, FeatureVector
from .labels import TrainingData, build_training_data


@dataclass
class RankedCandidate:
    candidate: Candidate
    score: float
    features: dict[str, float]


class TreatmentRanker:
    def __init__(self):
        self.booster: lgb.Booster | None = None

    def train(self, data: TrainingData, num_boost_round: int = 60) -> "TreatmentRanker":
        dataset = lgb.Dataset(data.X, label=np.array(data.y), group=np.array(data.groups),
                              feature_name=data.feature_names)
        params = {
            "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [3],
            "learning_rate": 0.1, "num_leaves": 7, "min_data_in_leaf": 1,
            "min_data_in_bin": 1, "max_depth": 4, "verbose": -1,
        }
        self.booster = lgb.train(params, dataset, num_boost_round=num_boost_round)
        return self

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or RANKER_MODEL_PATH)
        if self.booster is None:
            raise RuntimeError("no trained model to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))
        return path

    def load(self, path: Path | None = None) -> "TreatmentRanker":
        path = Path(path or RANKER_MODEL_PATH)
        if path.exists():
            self.booster = lgb.Booster(model_file=str(path))
        return self

    def _score_matrix(self, X: np.ndarray) -> np.ndarray:
        if self.booster is not None:
            return self.booster.predict(X)
        return X.sum(axis=1)

    def rank(self, candidates, features) -> list[RankedCandidate]:
        if not candidates:
            return []
        X = np.array([f.values for f in features], dtype=float)
        scores = self._score_matrix(X)
        ranked = [RankedCandidate(c, float(s), dict(zip(FEATURE_NAMES, f.values)))
                  for c, f, s in zip(candidates, features, scores)]
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked

    @property
    def is_trained(self) -> bool:
        return self.booster is not None

    def feature_importance(self) -> dict[str, float]:
        if self.booster is None:
            return {}
        imp = self.booster.feature_importance(importance_type="gain")
        return dict(zip(FEATURE_NAMES, [float(x) for x in imp]))


def train_and_save(source: str = "mock") -> tuple[TreatmentRanker, Path]:
    ranker = TreatmentRanker().train(build_training_data(source))
    return ranker, ranker.save()
