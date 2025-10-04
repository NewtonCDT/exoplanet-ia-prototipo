import argparse, json, os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

TARGETS = ["koi_disposition", "koi_pdisposition", "Disposition Using Kepler Data"]
FEATURE_SETS = [
    ["koi_period","koi_duration","koi_depth","koi_prad","koi_teq","koi_insol","koi_steff","koi_slogg","koi_smet"],
    ["Orbital Period [days]","Transit Duration [hrs]","Transit Depth [ppm]","Planetary Radius [Earth radii]",
     "Equilibrium Temperature [K]","Insolation Flux [Earth flux]","Stellar Effective Temperature [K]",
     "Stellar Surface Gravity [log10(cm/s**2)]","Stellar Metallicity [dex]"]
]
MAP = {"CONFIRMED":1,"Confirmed":1,"FALSE POSITIVE":0,"False Positive":0}

def pick_names(df):
    tgt = next((t for t in TARGETS if t in df.columns), None)
    if not tgt:
        raise SystemExit("No encuentro columna de disposición (koi_disposition / koi_pdisposition).")
    for feats in FEATURE_SETS:
        use = [c for c in feats if c in df.columns]
        if len(use) >= 5:
            return tgt, use
    raise SystemExit("No encuentro suficientes columnas de features esperadas.")

def load(csv_path):
    df = pd.read_csv(csv_path, comment="#")
    tgt, feats = pick_names(df)
    # Mantener solo CONFIRMED / FALSE POSITIVE
    m = df[tgt].isin(MAP.keys())
    df = df[m].copy()
    if df.empty:
        raise SystemExit(f"No quedan filas tras filtrar etiquetas válidas en {tgt}.")
    df["label"] = df[tgt].map(MAP)
    # Elimina filas con NaN en features
    df = df.dropna(subset=[*feats,"label"])
    # Chequeo de clases
    if df["label"].nunique() < 2:
        raise SystemExit(f"Solo hay una clase en {tgt}.")
    return df, feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="models/model.pkl")
    ap.add_argument("--metrics", default="reports/metrics.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)

    df, feats = load(args.csv)
    X, y = df[feats].values, df["label"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=200))])
    pipe.fit(Xtr, ytr)
    yp = pipe.predict(Xte); ypp = pipe.predict_proba(Xte)[:,1]
    metrics = {
        "features": feats,
        "accuracy": float(accuracy_score(yte, yp)),
        "precision": float(precision_score(yte, yp)),
        "recall": float(recall_score(yte, yp)),
        "f1": float(f1_score(yte, yp)),
        "roc_auc": float(roc_auc_score(yte, ypp)),
        "confusion_matrix": confusion_matrix(yte, yp).tolist()
    }
    joblib.dump({"pipeline": pipe, "features": feats}, args.out)
    with open(args.metrics, "w") as f: json.dump(metrics, f, indent=2)
    print("✅ Modelo guardado en", args.out)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
