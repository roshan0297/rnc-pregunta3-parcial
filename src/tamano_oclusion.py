"""Recall por tamano de caja y por nivel de oclusion (entregable 4).

Cada caja real se empareja con las predicciones de su imagen (IoU >= 0.5,
de mayor a menor confianza, una prediccion por caja). Se reporta:
    recall          = caja encontrada con la especie correcta
    localizada      = caja encontrada con cualquier especie (separa errores
                      de localizacion de errores de clasificacion)
Se usa recall y no AP por grupo porque un falso positivo no tiene tamano ni
oclusion "real" a la que asignarse.

Uso:
    python src/tamano_oclusion.py --pred runs/E3_principal/predicciones.csv \
        --oclusion results/oclusion/etiquetas_oclusion.csv --out runs/E3_principal/tamano_oclusion
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cct import cargar


def iou(a, b):
    """a: (4,), b: (n,4) en xyxy."""
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda c: (c[..., 2] - c[..., 0]) * (c[..., 3] - c[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def emparejar(gt, pred, umbral_iou=0.5):
    """Devuelve, por caja real, si fue encontrada con la especie correcta y con cualquiera."""
    gt = gt.copy()
    gt["recall"] = False
    gt["localizada"] = False
    pred_por_img = {k: g.sort_values("conf", ascending=False) for k, g in pred.groupby("file_name")}
    for nombre, g in gt.groupby("file_name"):
        p = pred_por_img.get(nombre)
        if p is None or p.empty:
            continue
        cajas_p = p[["x1", "y1", "x2", "y2"]].to_numpy()
        cajas_g = g[["gx1", "gy1", "gx2", "gy2"]].to_numpy()
        usada_ok = np.zeros(len(g), bool)
        for caja, esp in zip(cajas_p, p.especie):
            ious = iou(caja, cajas_g)
            gt.loc[g.index[ious >= umbral_iou], "localizada"] = True
            candidatas = np.where((ious >= umbral_iou) & (g.especie.to_numpy() == esp) & ~usada_ok)[0]
            if len(candidatas):
                j = candidatas[np.argmax(ious[candidatas])]
                usada_ok[j] = True
                gt.loc[g.index[j], "recall"] = True
    return gt


def kappa(a, b):
    """Kappa de Cohen entre dos etiquetados de las mismas cajas."""
    p_obs = (a.to_numpy() == b.to_numpy()).mean()
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    p_azar = sum(pa.get(c, 0) * pb.get(c, 0) for c in set(pa.index) | set(pb.index))
    return (p_obs - p_azar) / (1 - p_azar) if p_azar < 1 else 1.0


def resumen(gt, columna, orden):
    t = gt.groupby(["split", columna], observed=False).agg(n=("recall", "size"), recall=("recall", "mean"),
                                           localizada=("localizada", "mean"))
    return t.reindex(pd.MultiIndex.from_product([["cis_test", "trans_test"], orden]))


ETIQUETAS = {"pequena": "pequeña\n(<1 %)", "mediana": "mediana\n(1–5 %)", "grande": "grande\n(>5 %)",
             "bajo": "baja\n(<25 %)", "medio": "media\n(25–60 %)", "alto": "alta\n(>60 %)"}


def grafico(tablas, out):
    fig, axes = plt.subplots(1, len(tablas), figsize=(5 * len(tablas), 3.6), squeeze=False)
    for ax, (titulo, t) in zip(axes[0], tablas.items()):
        orden = list(dict.fromkeys(t.index.get_level_values(1)))   # unstack ordena alfabeticamente
        u = t.recall.unstack(0).reindex(orden)
        u.index = [ETIQUETAS.get(g, g) for g in u.index]
        u.plot.bar(ax=ax, color=["tab:green", "tab:red"], rot=0, fontsize=11)
        for k, fila in enumerate(t.n.unstack(0).reindex(orden).itertuples(index=False)):
            ax.text(k, 0.02, " / ".join(str(int(v)) if v == v else "0" for v in fila), ha="center", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(titulo.replace("Tamano", "Tamaño").replace("Oclusion", "Oclusión"), fontsize=12)
        ax.set_ylabel("recall (IoU ≥ 0,5, especie correcta)", fontsize=11)
        ax.set_xlabel("")
        ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "recall_tamano_oclusion.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--anot", default="data/raw/eccv_18_annotation_files")
    ap.add_argument("--oclusion", default="results/oclusion/etiquetas_oclusion.csv")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--cortes", type=float, nargs=2, default=[0.01, 0.05],
                    help="limites de area relativa pequena/mediana/grande")
    ap.add_argument("--excluir-clases", default="car")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    imagenes, cajas, _ = cargar(args.anot)
    excluir = set(args.excluir_clases.split(","))
    malas = set(cajas[cajas.especie.isin(excluir)].image_id)
    validas = set(imagenes[imagenes.estado == "con_caja"].id) - malas
    gt = cajas[cajas.split.isin(["cis_test", "trans_test"]) & cajas.image_id.isin(validas)].copy()
    gt["gx1"] = (gt.x / gt.img_w).clip(0, 1)
    gt["gy1"] = (gt.y / gt.img_h).clip(0, 1)
    gt["gx2"] = ((gt.x + gt.w) / gt.img_w).clip(0, 1)
    gt["gy2"] = ((gt.y + gt.h) / gt.img_h).clip(0, 1)

    pred = pd.read_csv(args.pred)
    pred = pred[(pred.especie.notna()) & (pred.especie != "") & (pred.conf >= args.conf)]
    gt = gt[gt.file_name.isin(pd.read_csv(args.pred).file_name)]  # solo imagenes que se predijeron
    gt = emparejar(gt, pred)

    a, b = args.cortes
    gt["tamano"] = pd.cut(gt.area_rel, [0, a, b, 1.01], labels=["pequena", "mediana", "grande"])
    tablas = {"Tamano relativo de la caja": resumen(gt, "tamano", ["pequena", "mediana", "grande"])}

    if Path(args.oclusion).exists():
        oc = pd.read_csv(args.oclusion)
        oc = oc[oc.nivel != "saltada"][["ann_id", "nivel"]]
        oc["ann_id"] = oc.ann_id.astype(str)
        gt["ann_id"] = gt.ann_id.astype(str)
        g_oc = gt.merge(oc, on="ann_id")
        tablas["Oclusion (muestra etiquetada)"] = resumen(g_oc, "nivel", ["bajo", "medio", "alto"])
        g_oc.to_csv(out / "cajas_oclusion_emparejadas.csv", index=False)

        # autoconsistencia del etiquetado: cajas re-etiquetadas a ciegas con --repetir
        rep = Path(args.oclusion).with_name("etiquetas_oclusion_repeticion.csv")
        if rep.exists():
            r = pd.read_csv(rep)
            r = r[r.nivel != "saltada"][["ann_id", "nivel"]].astype({"ann_id": str})
            par = oc.merge(r, on="ann_id", suffixes=("_1", "_2"))
            k = kappa(par.nivel_1, par.nivel_2)
            acuerdo = (par.nivel_1 == par.nivel_2).mean()
            print(f"\nautoconsistencia oclusion: n={len(par)}, acuerdo={acuerdo:.3f}, kappa={k:.3f}")
            pd.Series({"n": len(par), "acuerdo": acuerdo, "kappa": k}).to_csv(out / "kappa_oclusion.csv", header=False)
    else:
        print("sin archivo de oclusion:", args.oclusion)

    for titulo, t in tablas.items():
        print(f"\n{titulo} (conf >= {args.conf}):\n", t.round(3))
    pd.concat(tablas, names=["analisis"]).to_csv(out / "recall_por_grupo.csv")
    gt.to_csv(out / "cajas_emparejadas.csv", index=False)
    grafico(tablas, out)


if __name__ == "__main__":
    main()
