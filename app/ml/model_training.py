"""
ProcureLens Predictive Modeling & Risk Classification.

Trains, evaluates, and compares classification algorithms (Logistic Regression vs Random Forest)
to forecast payment delays on incoming procurement transactions.
Computes standard enterprise validation metrics (Precision, Recall, F1, ROC-AUC, Confusion Matrix).
"""

import json
import logging
import os
from typing import Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("procurelens.ml.model_training")


class ModelTrainer:
    """Trains and compares predictive risk models with rigorous evaluation metrics."""

    def __init__(self, risk_thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = risk_thresholds or {"low": 0.30, "high": 0.70}
        self.scaler = StandardScaler()
        self.models: Dict[str, object] = {}
        self.evaluations: Dict[str, Dict] = {}
        self.best_model_name: Optional[str] = None
        self.feature_names: list = []

    def prepare_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scales numeric features and records feature column names."""
        self.feature_names = list(X_train.columns)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def evaluate_model(
        self,
        model,
        X_test_scaled: np.ndarray,
        y_test: pd.Series,
        model_name: str
    ) -> Dict:
        """
        Calculates precision, recall, f1-score, ROC-AUC, and confusion matrix.
        """
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

        cm = confusion_matrix(y_test, y_pred)
        # [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # ROC-AUC calculation
        try:
            auc = round(float(roc_auc_score(y_test, y_prob)), 3)
        except Exception:
            auc = 0.5

        eval_result = {
            "model_name": model_name,
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 3),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 3),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 3),
            "roc_auc": auc,
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "raw_matrix": cm.tolist()
            },
            "test_sample_size": len(y_test),
            "delay_prevalence_in_test": round(float(y_test.mean()), 3)
        }
        self.evaluations[model_name] = eval_result
        return eval_result

    def train_logistic_regression(
        self,
        X_train_scaled: np.ndarray,
        y_train: pd.Series,
        X_test_scaled: np.ndarray,
        y_test: pd.Series
    ) -> Dict:
        """Trains Logistic Regression baseline with L2 penalty and balanced class weights."""
        lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
        lr.fit(X_train_scaled, y_train)
        self.models["Logistic Regression"] = lr
        return self.evaluate_model(lr, X_test_scaled, y_test, "Logistic Regression")

    def train_random_forest(
        self,
        X_train_scaled: np.ndarray,
        y_train: pd.Series,
        X_test_scaled: np.ndarray,
        y_test: pd.Series
    ) -> Dict:
        """Trains Random Forest Classifier with ensemble bagging."""
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42
        )
        rf.fit(X_train_scaled, y_train)
        self.models["Random Forest"] = rf
        return self.evaluate_model(rf, X_test_scaled, y_test, "Random Forest")

    def train_and_compare(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict:
        """Runs the entire training and benchmarking cycle across both models."""
        X_train_s, X_test_s = self.prepare_data(X_train, X_test)

        lr_eval = self.train_logistic_regression(X_train_s, y_train, X_test_s, y_test)
        rf_eval = self.train_random_forest(X_train_s, y_train, X_test_s, y_test)

        # Select model with higher ROC-AUC
        self.best_model_name = "Random Forest" if rf_eval["roc_auc"] >= lr_eval["roc_auc"] else "Logistic Regression"

        comparison = {
            "best_model": self.best_model_name,
            "models": {
                "Logistic Regression": lr_eval,
                "Random Forest": rf_eval
            },
            "feature_names": self.feature_names
        }

        # Feature importances if RF selected
        if "Random Forest" in self.models:
            rf = self.models["Random Forest"]
            importances = dict(zip(self.feature_names, rf.feature_importances_))
            sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
            comparison["feature_importances"] = {k: round(float(v), 4) for k, v in sorted_imp.items()}

        logger.info(f"Model benchmarking complete. Best model: {self.best_model_name}")
        return comparison

    def save_artifacts(self, artifact_dir: str = "models") -> str:
        """Persists trained model, scaler, and metadata to disk."""
        os.makedirs(artifact_dir, exist_ok=True)
        model_path = os.path.join(artifact_dir, "procurelens_risk_model.joblib")

        best_model = self.models.get(self.best_model_name or "Random Forest")
        payload = {
            "model": best_model,
            "model_name": self.best_model_name,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "thresholds": self.thresholds,
            "evaluations": self.evaluations
        }

        joblib.dump(payload, model_path)
        logger.info(f"Saved model artifact to {model_path}")
        return model_path
