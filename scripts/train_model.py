"""
Train and persist predictive risk models for ProcureLens.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.connection import get_engine
from app.database.queries import (
    get_all_event_logs,
    get_all_invoices,
    get_all_payments,
    get_all_purchase_orders,
    get_all_suppliers,
)
from app.ml.feature_engineering import FeatureEngineer
from app.ml.model_training import ModelTrainer


def main():
    print("Loading procurement dataset for ML training...")
    engine = get_engine()
    pos = get_all_purchase_orders(engine)
    invs = get_all_invoices(engine)
    pays = get_all_payments(engine)
    evts = get_all_event_logs(engine)
    sups = get_all_suppliers(engine)

    print("Building leakage-safe features...")
    fe = FeatureEngineer()
    features_df = fe.build_features(pos, invs, pays, evts, sups)
    print(f"Engineered feature matrix: {features_df.shape[0]} rows x {features_df.shape[1]} columns")

    X_train, X_test, y_train, y_test = fe.get_time_split(features_df, split_month=4)
    print(f"Train split: {len(X_train)} samples | Test split: {len(X_test)} samples")

    trainer = ModelTrainer()
    results = trainer.train_and_compare(X_train, X_test, y_train, y_test)

    print("\n================ Model Evaluation ================")
    print(f"Selected Best Model: {results['best_model']}")
    for model_name, ev in results["models"].items():
        print(f"\nModel: {model_name}")
        print(f"  ROC-AUC:   {ev['roc_auc']:.3f}")
        print(f"  F1-Score:  {ev['f1_score']:.3f}")
        print(f"  Precision: {ev['precision']:.3f}")
        print(f"  Recall:    {ev['recall']:.3f}")
        cm = ev["confusion_matrix"]
        print(f"  Confusion Matrix: TN={cm['true_negative']}, FP={cm['false_positive']}, FN={cm['false_negative']}, TP={cm['true_positive']}")

    print("\nTop Feature Importances:")
    for feat, imp in list(results.get("feature_importances", {}).items())[:6]:
        print(f"  - {feat:<28} {imp:.4f}")

    path = trainer.save_artifacts()
    print(f"\nModel artifacts successfully saved to: {path}")


if __name__ == "__main__":
    main()
