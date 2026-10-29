"""Exploracion de CCT-20: tablas y figuras que van al informe (Fase 1).

Uso:
    python src/explorar.py                      # solo anotaciones
    python src/explorar.py --imgs data/raw/imagenes   # ademas: noche/dia y mosaicos
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from cct import SPLITS, cargar


def tabla_especies(imagenes, cajas, categorias, out):
    """Imagenes y cajas por especie y split, mas las imagenes sin caja."""
    filas = []
    for cid, nombre in sorted(categorias.items(), key=lambda t: t[1]):
        fila = {"especie": nombre}
        for s in SPLITS:
            im_s = imagenes[imagenes.split == s]
            fila[f"{s}_imgs"] = im_s.categorias.apply(lambda c: cid in c).sum()
            fila[f"{s}_cajas"] = ((cajas.split == s) & (cajas.category_id == cid)).sum()
            fila[f"{s}_sin_caja"] = (im_s.categorias.apply(lambda c: cid in c) & (im_s.estado == "sin_caja")).sum()
        filas.append(fila)
    t = pd.DataFrame(filas).sort_values("train_cajas", ascending=False)
    t.to_csv(out / "tabla_especies.csv", index=False)

    resumen = imagenes.groupby(["split", "estado"]).size().unstack(fill_value=0).reindex(SPLITS)
    resumen["total"] = resumen.sum(axis=1)
    resumen.to_csv(out / "resumen_estados.csv")
    print("\nImagenes por split y estado:\n", resumen)
    print("\nCajas por especie (train / cis_test / trans_test):")
    print(t[["especie", "train_cajas", "cis_test_cajas", "trans_test_cajas"]].to_string(index=False))
    return t


def ubicaciones(imagenes, categorias, out):
    locs = {s: set(imagenes[imagenes.split == s].location) for s in SPLITS}
    print("\nUbicaciones por split:")
    for s in SPLITS:
        print(f"  {s:10s} {len(locs[s]):3d}  {sorted(locs[s], key=int)}")
    print("  cruce train vs trans_test:", locs["train"] & locs["trans_test"] or "ninguno")

    # cuanto de cada especie hay en cada camara de train: base del "atajo por fondo"
    tr = imagenes[(imagenes.split == "train") & (imagenes.estado != "vacia")]
    tr = tr.explode("categorias").reset_index(drop=True)
    pivote = pd.crosstab(tr.location, tr.categorias.map(categorias), normalize="index").round(3)
    pivote.to_csv(out / "especies_por_ubicacion_train.csv")
    dominante = pivote.max(axis=1)
    print("\nFraccion de la especie dominante por camara (train):")
    print(dominante.sort_values(ascending=False).to_string())


def histograma_cajas(cajas, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = np.logspace(-5, 0, 40)
    for s, c in [("train", "tab:blue"), ("cis_test", "tab:green"), ("trans_test", "tab:red")]:
        axes[0].hist(cajas[cajas.split == s].area_rel, bins=bins, histtype="step", lw=2, color=c,
                     label=s, density=True)
    for corte in (0.01, 0.05):
        axes[0].axvline(corte, ls="--", c="gray")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("area de la caja / area de la imagen")
    axes[0].set_ylabel("densidad")
    axes[0].legend()

    tr = cajas[cajas.split == "train"]
    orden = tr.groupby("especie").area_rel.median().sort_values().index
    axes[1].boxplot([tr[tr.especie == e].area_rel for e in orden], tick_labels=orden, vert=False,
                    showfliers=False)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("area relativa (train)")
    fig.tight_layout()
    fig.savefig(out / "tamano_cajas.png", dpi=150)

    t = cajas.groupby("split").area_rel.describe(percentiles=[.1, .5, .9])
    t["pct_menor_1pct"] = cajas.groupby("split").area_rel.apply(lambda a: (a < 0.01).mean())
    t.to_csv(out / "tamano_cajas.csv")
    print("\nArea relativa de las cajas:\n", t.round(4))


def horas(imagenes, out):
    fig, ax = plt.subplots(figsize=(7, 3))
    for s, c in [("train", "tab:blue"), ("trans_test", "tab:red")]:
        h = imagenes[(imagenes.split == s) & (imagenes.estado != "vacia")].date_captured.dt.hour
        ax.hist(h, bins=24, range=(0, 24), histtype="step", lw=2, color=c, label=s, density=True)
    ax.set_xlabel("hora de captura")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "hora_captura.png", dpi=150)


def indice_imagenes(dir_imgs):
    return {p.name: p for p in Path(dir_imgs).rglob("*.jpg")}


def es_noche(ruta):
    """Las fotos infrarrojas nocturnas son practicamente grises (R~G~B)."""
    with Image.open(ruta) as im:
        a = np.asarray(im.convert("RGB").resize((128, 96)), dtype=np.float32)
    return np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean() < 4


def noche_dia(imagenes, rutas, out, n=600, semilla=0):
    filas = []
    for s in SPLITS:
        m = imagenes[(imagenes.split == s) & imagenes.file_name.isin(rutas)]
        m = m.sample(min(n, len(m)), random_state=semilla)
        filas.append({"split": s, "n_muestra": len(m),
                      "pct_noche": np.mean([es_noche(rutas[f]) for f in m.file_name])})
    t = pd.DataFrame(filas)
    t.to_csv(out / "noche_dia.csv", index=False)
    print("\nFraccion de fotos nocturnas (muestra):\n", t)


def mosaico(imagenes, cajas, rutas, out, por_especie=4, semilla=0):
    """Ejemplos aleatorios por especie para elegir a mano los casos dificiles del informe."""
    rng = random.Random(semilla)
    tr = cajas[cajas.split == "train"]
    especies = tr.especie.value_counts().index
    fig, axes = plt.subplots(len(especies), por_especie, figsize=(3 * por_especie, 2.3 * len(especies)))
    info = imagenes.set_index("id")
    for i, esp in enumerate(especies):
        ids = list(tr[tr.especie == esp].image_id.unique())
        rng.shuffle(ids)
        for j in range(por_especie):
            ax = axes[i, j]
            ax.axis("off")
            if j >= len(ids) or info.loc[ids[j], "file_name"] not in rutas:
                continue
            fila = info.loc[ids[j]]
            with Image.open(rutas[fila.file_name]) as im:
                esc = im.width / fila.width  # las imagenes "sm" estan reducidas
                ax.imshow(im)
            for _, c in tr[tr.image_id == ids[j]].iterrows():
                ax.add_patch(patches.Rectangle((c.x * esc, c.y * esc), c.w * esc, c.h * esc,
                                               fill=False, ec="yellow", lw=1.5))
            ax.set_title(f"{esp} | loc {fila.location}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "mosaico_especies.png", dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anot", default="data/raw/eccv_18_annotation_files")
    ap.add_argument("--imgs", default=None, help="carpeta con las imagenes (opcional)")
    ap.add_argument("--out", default="results/exploracion")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    imagenes, cajas, categorias = cargar(args.anot)

    tabla_especies(imagenes, cajas, categorias, out)
    ubicaciones(imagenes, categorias, out)
    histograma_cajas(cajas, out)
    horas(imagenes, out)

    if args.imgs:
        rutas = indice_imagenes(args.imgs)
        print(f"\n{len(rutas)} imagenes encontradas en {args.imgs}")
        noche_dia(imagenes, rutas, out)
        mosaico(imagenes, cajas, rutas, out)
    print(f"\nResultados en {out}/")


if __name__ == "__main__":
    main()
