set -euo pipefail

PROJECT=~/spaceapps/exoplanet-ia-prototipo
RAW=$PROJECT/data/raw
PROCESSED=$PROJECT/data/processed
MODELS=$PROJECT/models
REPORTS=$PROJECT/reports
SRC=$PROJECT/src

echo "==> Preparando estructura limpia"
rm -rf "$PROJECT/.venv" "$PROCESSED" "$MODELS" "$REPORTS" "$SRC"
mkdir -p "$RAW" "$PROCESSED" "$MODELS" "$REPORTS" "$SRC"

echo "==> Creando requirements.txt"
cat > $PROJECT/requirements.txt <<'EOF'
pandas>=2.0
numpy>=1.26
scikit-learn>=1.4
matplotlib>=3.8
joblib>=1.3
EOF

echo "==> Escribiendo src/train.py"
cat > $SRC/train.py <<'EOF'
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
EOF

echo "==> Escribiendo src/infer.py"
cat > $SRC/infer.py <<'EOF'
import argparse, os, joblib, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--out", default="data/processed/predicciones.csv")
args = ap.parse_args()

bundle = joblib.load(args.model)
pipe, feats = bundle["pipeline"], bundle["features"]
df = pd.read_csv(args.csv, comment="#")

missing = [c for c in feats if c not in df.columns]
if missing:
    raise SystemExit(f"Faltan columnas requeridas en el CSV: {missing}")

mask = df[feats].notna().all(axis=1)
kept, skipped = int(mask.sum()), int((~mask).sum())
if kept == 0:
    raise SystemExit("Todas las filas tienen NaN en las features.")
df_valid = df.loc[mask].copy()

proba = pipe.predict_proba(df_valid[feats])[:,1]
pred = pipe.predict(df_valid[feats])
out = df_valid.copy()
out["pred_label"] = pred
out["pred_proba_confirmed"] = proba

os.makedirs(os.path.dirname(args.out), exist_ok=True)
out.to_csv(args.out, index=False)
print(f"✅ Predicciones guardadas en {args.out}")
print(f"Filas predichas: {kept} | Filas omitidas por NaN: {skipped}")
EOF

echo "==> Creando entorno virtual e instalando dependencias"
python3 -m venv $PROJECT/.venv
source $PROJECT/.venv/bin/activate
python -m pip install --upgrade pip
pip install -r $PROJECT/requirements.txt

echo "==> Verificando que exista el CSV de la NASA"
if [ ! -f "$RAW/kepler_koi_full.csv" ]; then
  echo "❌ Falta $RAW/kepler_koi_full.csv"
  echo "Copia aquí tu CSV del KOI Cumulative antes de continuar."
  exit 1
fi

echo "==> Entrenando"
python $SRC/train.py --csv $RAW/kepler_koi_full.csv --out $MODELS/model.pkl --metrics $REPORTS/metrics.json

echo "==> Inferencia"
python $SRC/infer.py --model $MODELS/model.pkl --csv $RAW/kepler_koi_full.csv --out $PROCESSED/predicciones.csv

echo "==> Gráfico (distribución de probabilidades)"
python - <<'PY'
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
df = pd.read_csv("data/processed/predicciones.csv")
plt.hist(df["pred_proba_confirmed"], bins=30)
plt.xlabel("Probabilidad de ser planeta confirmado")
plt.ylabel("Cantidad de objetos")
plt.title("Distribución de probabilidades - Modelo IA SpaceApps 2025")
plt.tight_layout()
plt.savefig("reports/grafico_predicciones.png")
print("✅ Gráfico guardado en reports/grafico_predicciones.png")
PY

echo "==> Top 10 candidatos por probabilidad"
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/processed/predicciones.csv")
cols = [c for c in ["kepoi_name","kepler_name","koi_disposition","koi_period","koi_prad","koi_steff","pred_proba_confirmed"] if c in df.columns]
print(df.sort_values("pred_proba_confirmed", ascending=False)[cols].head(10).to_string(index=False))
PY

echo "==> Listo."
