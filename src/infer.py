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
