"""AP por especie vs numero de cajas de entrenamiento (entregable 2).

Uso:
    python src/analisis_desbalance.py --eval runs/E3_principal/eval_por_especie.csv \
        --yolo data/yolo_vacias20 --out runs/E3_principal/desbalance
"""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def cajas_train(dir_yolo):
    nombres = yaml.safe_load(open(dir_yolo / "cct.yaml"))["names"]
    cuenta = Counter()
    for txt in (dir_yolo / "labels" / "train").glob("*.txt"):
        for linea in txt.read_text().splitlines():
            cuenta[nombres[int(linea.split()[0])]] += 1
    return pd.Series(cuenta, name="cajas_train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ev = pd.read_csv(args.eval)
    ev = ev[ev.especie != "TODAS (mAP)"]
    n = cajas_train(Path(args.yolo))
    t = ev.pivot(index="especie", columns="test", values="AP50").join(n).sort_values("cajas_train")
    t["brecha_cis_trans"] = t["cis_test"] - t["trans_test"]
    t.to_csv(out / "ap_vs_cajas_train.csv")
    print(t.round(3))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for test, color in [("cis_test", "tab:green"), ("trans_test", "tab:red")]:
        ax.scatter(t.cajas_train, t[test], c=color, label=test, s=40)
    for esp, fila in t.iterrows():
        ax.annotate(esp, (fila.cajas_train, fila.cis_test), fontsize=8, xytext=(4, 2), textcoords="offset points")
    # tendencia en escala log solo como guia visual
    for test, color in [("cis_test", "tab:green"), ("trans_test", "tab:red")]:
        ok = t[test].notna() & (t.cajas_train > 0)
        if ok.sum() > 2:
            p = np.polyfit(np.log10(t.cajas_train[ok]), t[test][ok], 1)
            xs = np.logspace(np.log10(t.cajas_train[ok].min()), np.log10(t.cajas_train[ok].max()), 50)
            ax.plot(xs, np.polyval(p, np.log10(xs)), c=color, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("cajas de entrenamiento (escala log)")
    ax.set_ylabel("AP@0.5")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "ap_vs_cajas_train.png", dpi=150)

    corr = t[["cajas_train", "cis_test", "trans_test"]].corr(method="spearman").loc["cajas_train"]
    print("\ncorrelacion de Spearman con el n de cajas:\n", corr.round(3))


if __name__ == "__main__":
    main()
