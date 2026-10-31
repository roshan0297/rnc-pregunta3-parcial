"""Analisis complementarios citados en el informe (secciones 5.2-5.4).

No necesita GPU: trabaja sobre las anotaciones y los CSV que dejan los demas scripts.
    1. Spearman entre AP@0.5 y cajas de entrenamiento, con p-valor.
    2. Concentracion de cada especie en las camaras de train frente a su caida cis -> trans.
    3. Cajas grandes de trans_test: borde de la imagen, hora nocturna y especie confundida.
    4. Acuerdo del etiquetado de oclusion (tal cual y leyendo la escala humana como visibilidad).

Uso:
    python src/verificaciones.py --pred runs/E3_principal/predicciones.csv \
        --cajas runs/E3_principal/tamano_oclusion/cajas_emparejadas.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cct import cargar
from tamano_oclusion import iou, kappa

NOCHE = list(range(19, 24)) + list(range(0, 6))


def desbalance(ap):
    filas = []
    for test in ["cis_test", "trans_test"]:
        ok = ap[test].notna()
        r = spearmanr(ap.cajas_train[ok], ap[test][ok])
        filas.append({"test": test, "especies": int(ok.sum()), "rho": r.statistic, "p": r.pvalue})
    return pd.DataFrame(filas)


def concentracion(cajas, imagenes, ap):
    t = cajas[cajas.split == "train"].merge(imagenes[["id", "location"]], left_on="image_id", right_on="id")
    ct = pd.crosstab(t.especie, t.location)
    fr = ct.div(ct.sum(axis=1), axis=0)
    c = pd.DataFrame({"cajas_train": ct.sum(axis=1), "camaras": (ct > 0).sum(axis=1),
                      "frac_camara_principal": fr.max(axis=1), "camara_principal": fr.idxmax(axis=1),
                      "frac_2_principales": fr.apply(lambda r: r.nlargest(2).sum(), axis=1)})
    c = c.join(ap["brecha_cis_trans"])
    # especies con datos suficientes en ambos tests (fuera car, zorro, tejon y ciervo)
    ok = c.drop(index=["car", "fox", "badger", "deer"], errors="ignore").dropna(subset=["brecha_cis_trans"])
    r = spearmanr(ok.frac_2_principales, ok.brecha_cis_trans)
    return c.sort_values("frac_2_principales", ascending=False), len(ok), r.statistic, r.pvalue


def cajas_grandes(g, pred, imagenes):
    g = g.merge(imagenes[["file_name", "date_captured"]], on="file_name", how="left")
    g["noche"] = pd.to_datetime(g.date_captured).dt.hour.isin(NOCHE)
    g["borde"] = (g.gx1 < 0.01) | (g.gy1 < 0.01) | (g.gx2 > 0.99) | (g.gy2 > 0.99)
    t = g[(g.split == "trans_test") & (g.tamano == "grande")].copy()
    t["grupo"] = np.where(t.recall, "acierto", np.where(t.localizada, "especie_erronea", "no_localizada"))
    resumen = t.groupby("grupo").agg(n=("recall", "size"), toca_borde=("borde", "mean"),
                                     nocturna=("noche", "mean"), area_mediana=("area_rel", "median"))
    p = pred[pred.especie.notna() & (pred.especie != "") & (pred.conf >= 0.5)]
    por_img = {k: v for k, v in p.groupby("file_name")}
    pares = []
    for r in t[t.grupo == "especie_erronea"].itertuples():
        q = por_img.get(r.file_name)
        if q is None:
            continue
        j = iou(np.array([r.gx1, r.gy1, r.gx2, r.gy2]), q[["x1", "y1", "x2", "y2"]].to_numpy()).argmax()
        pares.append((r.especie, q.especie.iloc[j]))
    confusiones = pd.DataFrame(pares, columns=["real", "predicha"]).value_counts().rename("cajas").reset_index()
    return resumen, confusiones


def acuerdo_oclusion(dir_ocl):
    a = pd.read_csv(dir_ocl / "etiquetas_oclusion.csv", dtype={"ann_id": str})
    b = pd.read_csv(dir_ocl / "etiquetas_oclusion_repeticion.csv", dtype={"ann_id": str})
    m = a.merge(b[["ann_id", "nivel"]], on="ann_id", suffixes=("_modelo", "_humano"))
    m = m[(m.nivel_modelo != "saltada") & (m.nivel_humano != "saltada")]
    invertida = m.nivel_humano.map({"alto": "bajo", "bajo": "alto", "medio": "medio"})
    res = pd.DataFrame([
        {"lectura": "tal cual", "n": len(m), "acuerdo": (m.nivel_modelo == m.nivel_humano).mean(),
         "kappa": kappa(m.nivel_modelo, m.nivel_humano)},
        {"lectura": "humano como visibilidad", "n": len(m), "acuerdo": (m.nivel_modelo == invertida).mean(),
         "kappa": kappa(m.nivel_modelo, invertida)}])
    return res, pd.crosstab(m.nivel_modelo, m.nivel_humano)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--anot", default="data/raw/eccv_18_annotation_files")
    ap_.add_argument("--pred", required=True, help="CSV de predecir.py")
    ap_.add_argument("--cajas", required=True, help="cajas_emparejadas.csv de tamano_oclusion.py")
    ap_.add_argument("--ap", default="results/desbalance/ap_vs_cajas_train.csv", help="CSV de analisis_desbalance.py")
    ap_.add_argument("--oclusion", default="results/oclusion", help="carpeta con las etiquetas de oclusion")
    ap_.add_argument("--out", default="results/verificaciones")
    args = ap_.parse_args()
    ocl, out = Path(args.oclusion), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    imagenes, cajas, _ = cargar(args.anot)
    ap = pd.read_csv(args.ap, index_col="especie")

    d = desbalance(ap)
    d.to_csv(out / "spearman_ap_vs_cajas.csv", index=False)
    print("1. AP@0.5 vs cajas de entrenamiento (Spearman):\n", d.round(3).to_string(index=False))

    c, n, rho, p = concentracion(cajas, imagenes, ap)
    c.to_csv(out / "concentracion_por_camara.csv")
    print("\n2. Concentracion en camaras de train:\n", c.round(2).to_string())
    print(f"   Spearman concentracion vs caida ({n} especies): rho={rho:.2f}, p={p:.2f}")

    resumen, conf = cajas_grandes(pd.read_csv(args.cajas), pd.read_csv(args.pred), imagenes)
    resumen.to_csv(out / "cajas_grandes_trans.csv")
    conf.to_csv(out / "confusiones_cajas_grandes_trans.csv", index=False)
    print("\n3. Cajas grandes de trans_test:\n", resumen.round(3).to_string())
    print(" confusiones mas frecuentes:\n", conf.head(6).to_string(index=False))

    if (ocl / "etiquetas_oclusion_repeticion.csv").exists():
        a, tabla = acuerdo_oclusion(ocl)
        a.to_csv(out / "acuerdo_oclusion.csv", index=False)
        tabla.to_csv(out / "acuerdo_oclusion_tabla.csv")
        print("\n4. Acuerdo del etiquetado de oclusion:\n", a.round(3).to_string(index=False))
        print(tabla.to_string())
    print("\nresultados en", out)


if __name__ == "__main__":
    main()
