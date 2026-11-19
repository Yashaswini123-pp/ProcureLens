"""
ProcureLens Inference & Operational Risk Predictor.

Applies trained risk classification models to score new procurement transactions.
Assigns tri-tier risk levels (LOW, MEDIUM, HIGH) and generates interpretability insights.
"""

import logging
import os
from typing import Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("procurelens.ml.predictor")


class PaymentDelayPredictor:
    """Inference service for assessing invoice payment delay risks."""

    def __init__(self, model_artifact_path: Optional[str] = None):
        self.model_path = model_artifact_path or "models/procurelens_risk_model.joblib"
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.thresholds = {"low": 0.30, "high": 0.70}
        self.is_loaded = False
        self._load_artifact()

    def _load_artifact(self) -> None:
        """Loads serialized model and preprocessing transformers."""
        if os.path.exists(self.model_path):
            try:
                payload = joblib.load(self.model_path)
                self.model = payload["model"]
                self.scaler = payload["scaler"]
                self.feature_names = payload["feature_names"]
                self.thresholds = payload.get("thresholds", self.thresholds)
                self.is_loaded = True
                logger.info(f"Loaded ML model '{payload.get('model_name')}' from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load ML artifact from {self.model_path}: {str(e)}")
                self.is_loaded = False
        else:
            logger.info(f"ML artifact not found at {self.model_path}. Predictor will run in heuristic fallback mode.")

    def get_risk_category(self, probability: float) -> str:
        """
        Classifies risk probability into tri-tier operational category:
        - LOW: < low_threshold (default 0.30)
        - MEDIUM: low_threshold to < high_threshold (default 0.30 to 0.70)
        - HIGH: >= high_threshold (default 0.70)
        """
        if probability >= self.thresholds["high"]:
            return "HIGH"
        elif probability >= self.thresholds["low"]:
            return "MEDIUM"
        return "LOW"

    def predict_single(self, features_dict: Dict) -> Dict:
        """
        Calculates delay probability and risk tier for a single transaction.
        """
        if not self.is_loaded or self.model is None:
            # Domain heuristic fallback if no model trained yet
            amt = float(features_dict.get("purchase_amount", 5000))
            rating = float(features_dict.get("supplier_rating", 3.5))
            rework = int(features_dict.get("has_rework", 0))

            base_prob = 0.12  # baseline synthetic delay rate
            if amt > 100000:
                base_prob += 0.25
            if rating < 2.5:
                base_prob += 0.30
            if rework:
                base_prob += 0.20
            prob = min(max(base_prob, 0.05), 0.95)

            return {
                "probability": round(prob, 3),
                "risk_category": self.get_risk_category(prob),
                "is_model_trained": False,
                "notes": "Estimated via heuristic rules. Train ML model for statistical inference."
            }

        # Vectorize input using expected feature sequence
        row = [float(features_dict.get(col, 0)) for col in self.feature_names]
        scaled = self.scaler.transform([row])
        prob = float(self.model.predict_proba(scaled)[0, 1])

        return {
            "probability": round(prob, 3),
            "risk_category": self.get_risk_category(prob),
            "is_model_trained": True,
            "thresholds": self.thresholds
        }

    def predict_batch(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Scores an entire DataFrame of procurement transactions.
        """
        result_df = df_features.copy()

        if not self.is_loaded or self.model is None:
            # Fallback heuristic
            probs = []
            for _, r in result_df.iterrows():
                amt = float(r.get("purchase_amount", 5000))
                rating = float(r.get("supplier_rating", 3.5))
                p = 0.12 + (0.25 if amt > 100000 else 0) + (0.30 if rating < 2.5 else 0)
                probs.append(min(max(p, 0.05), 0.95))
            result_df["delay_probability"] = [round(p, 3) for p in probs]
            result_df["risk_category"] = [self.get_risk_category(p) for p in probs]
            return result_df

        # Match columns
        X = result_df[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        result_df["delay_probability"] = [round(float(p), 3) for p in probabilities]
        result_df["risk_category"] = [self.get_risk_category(p) for p in probabilities]
        return result_df

    @staticmethod
    def get_model_limitations() -> List[str]:
        """Returns standard enterprise governance notes regarding model boundaries."""
        return [
            "Trained exclusively on historical Purchase-to-Pay transactional patterns.",
            "Prediction horizon is limited to invoice settlement lifecycle; external macroeconomic shocks are unmodeled.",
            "Features strictly omit all post-invoice payment attributes to guarantee zero data leakage.",
            "Predictions are risk indicators to assist finance prioritization, not automated payment approval decisions."
        ]
