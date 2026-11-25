import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def build_and_save_model(model_path: str = "clinical_risk_model.pkl"):
    """Trains a Random Forest classifier on key blood biomarkers and saves it as a .pkl artifact."""
    rng = np.random.default_rng(42)
    n_samples = 1500

    # Clinical dataset modeled on standard diagnostic reference ranges
    hemoglobin = rng.uniform(7.5, 18.0, n_samples)     # Normal: 12.0 - 17.0 g/dL
    mcv = rng.uniform(60.0, 110.0, n_samples)          # Normal: 80 - 100 fL
    wbc = rng.uniform(2500, 18000, n_samples)          # Normal: 4000 - 11000 /uL
    platelets = rng.uniform(80000, 550000, n_samples)  # Normal: 150000 - 450000 /uL
    hba1c = rng.uniform(4.2, 11.5, n_samples)          # Normal: < 5.7 %
    ldl = rng.uniform(60.0, 220.0, n_samples)          # Normal: < 100 mg/dL

    labels = []
    for hb, m, w, p, a1c, l in zip(hemoglobin, mcv, wbc, platelets, hba1c, ldl):
        # 2 = High Priority (Consult Doctor Soon), 1 = Borderline / Monitor, 0 = Normal Range
        if hb < 10.5 or w > 14000 or w < 3200 or p < 120000 or a1c >= 6.5 or l >= 160:
            labels.append(2)
        elif hb < 12.5 or m < 80 or m > 100 or a1c >= 5.7 or l >= 130:
            labels.append(1)
        else:
            labels.append(0)

    df = pd.DataFrame({
        "hemoglobin": hemoglobin,
        "mcv": mcv,
        "wbc": wbc,
        "platelets": platelets,
        "hba1c": hba1c,
        "ldl": ldl,
        "risk_class": labels,
    })

    feature_cols = ["hemoglobin", "mcv", "wbc", "platelets", "hba1c", "ldl"]
    X = df[feature_cols]
    y = df["risk_class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    acc = round(accuracy_score(y_test, clf.predict(X_test)) * 100, 2)
    artifact = {
        "model": clf,
        "features": feature_cols,
        "accuracy": acc,
        "samples": n_samples,
    }
    joblib.dump(artifact, model_path)
    return artifact


if __name__ == "__main__":
    info = build_and_save_model()
    print(f"Saved clinical_risk_model.pkl | Test Accuracy: {info['accuracy']}%")
