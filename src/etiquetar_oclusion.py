"""Herramienta para etiquetar a mano el nivel de oclusion de cajas de test.

CCT-20 no trae etiqueta de oclusion, asi que se crea una muestra propia.
Corre en tu PC (no necesita las 6 GB): baja solo las imagenes de la muestra
desde el bucket publico de LILA, en resolucion original.

Criterio (porcentaje del animal que NO se ve, por vegetacion, otro objeto o el borde):
    1 = bajo   (< 25 %)
    2 = medio  (25-60 %)
    3 = alto   (> 60 %)
Teclas: 1 / 2 / 3, s = saltar (caja mal anotada o imposible de juzgar), q = salir.
Se guarda despues de cada tecla; al volver a correrlo continua donde quedo.

Uso:
    python src/etiquetar_oclusion.py                 # 150 cajas de cis_test + 150 de trans_test
    python src/etiquetar_oclusion.py --repetir 50    # re-etiquetar 50 a ciegas (consistencia)
"""
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from cct import cargar

URL_IMG_FULL = "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/{}"
NIVELES = {"1": "bajo", "2": "medio", "3": "alto", "s": "saltada"}


def muestra(anot, n_por_split, excluir, semilla):
    imagenes, cajas, _ = cargar(anot)
    validas = imagenes[imagenes.estado == "con_caja"].id
    c = cajas[cajas.split.isin(["cis_test", "trans_test"]) & cajas.image_id.isin(validas)
              & ~cajas.especie.isin(excluir)]
    partes = [g.sample(min(n_por_split, len(g)), random_state=semilla) for _, g in c.groupby("split")]
    # mezclar cis y trans para no sesgar el criterio
    return pd.concat(partes).sample(frac=1, random_state=semilla)


def bajar(nombre, carpeta):
    destino = carpeta / nombre
    if not destino.exists():
        try:
            urllib.request.urlretrieve(URL_IMG_FULL.format(nombre), destino)
        except Exception:
            return None
    return destino


def etiquetar(cajas, carpeta, salida, mostrar_previa):
    pendientes = list(cajas.itertuples())
    estado = {"i": 0}
    fig, (ax_img, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.6, 1]})

    def guardar(nivel):
        c = pendientes[estado["i"]]
        fila = pd.DataFrame([{"ann_id": c.ann_id, "file_name": c.file_name, "split": c.split,
                              "especie": c.especie, "area_rel": c.area_rel, "nivel": NIVELES[nivel]}])
        fila.to_csv(salida, mode="a", header=not salida.exists(), index=False)

    def mostrar():
        while estado["i"] < len(pendientes):
            c = pendientes[estado["i"]]
            ruta = carpeta / c.file_name
            if ruta.exists():
                break
            estado["i"] += 1  # imagen no disponible en el bucket
        else:
            plt.close(fig)
            return
        with Image.open(ruta) as im:
            im = im.convert("RGB")
            m = max(c.w, c.h) * 0.6
            zoom = im.crop((max(c.x - m, 0), max(c.y - m, 0),
                            min(c.x + c.w + m, im.width), min(c.y + c.h + m, im.height)))
            ax_img.clear(), ax_zoom.clear()
            ax_img.imshow(im)
        ax_img.add_patch(patches.Rectangle((c.x, c.y), c.w, c.h, fill=False, ec="yellow", lw=2))
        ax_zoom.imshow(zoom)
        ax_zoom.add_patch(patches.Rectangle((min(m, c.x), min(m, c.y)), c.w, c.h, fill=False, ec="yellow", lw=1.5))
        for ax in (ax_img, ax_zoom):
            ax.axis("off")
        titulo = f"[{estado['i'] + 1}/{len(pendientes)}] {c.especie}"
        if mostrar_previa:
            titulo += f" ({c.split})"
        fig.suptitle(titulo + "    1=bajo <25%   2=medio 25-60%   3=alto >60%   s=saltar   q=salir")
        fig.canvas.draw_idle()

    def tecla(ev):
        if ev.key == "q":
            plt.close(fig)
        elif ev.key in NIVELES:
            guardar(ev.key)
            estado["i"] += 1
            mostrar()

    fig.canvas.mpl_connect("key_press_event", tecla)
    mostrar()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anot", default="data/raw/eccv_18_annotation_files")
    ap.add_argument("--n", type=int, default=150, help="cajas por split")
    ap.add_argument("--excluir-clases", default="car")
    ap.add_argument("--salida", default="results/oclusion/etiquetas_oclusion.csv")
    ap.add_argument("--repetir", type=int, default=0,
                    help="re-etiquetar N cajas ya hechas, sin ver la etiqueta anterior")
    ap.add_argument("--semilla", type=int, default=7)
    args = ap.parse_args()

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    carpeta = Path("data/oclusion_imgs")
    carpeta.mkdir(parents=True, exist_ok=True)

    cajas = muestra(args.anot, args.n, set(args.excluir_clases.split(",")), args.semilla)
    if args.repetir:
        hechas = pd.read_csv(salida)
        hechas = hechas[hechas.nivel != "saltada"].sample(args.repetir, random_state=args.semilla + 1)
        todas = muestra(args.anot, 10**6, set(), 0)
        cajas = todas[todas.ann_id.astype(str).isin(hechas.ann_id.astype(str))]
        salida = salida.with_name("etiquetas_oclusion_repeticion.csv")
    if salida.exists():
        ya = set(pd.read_csv(salida).ann_id.astype(str))
        cajas = cajas[~cajas.ann_id.astype(str).isin(ya)]
    print(f"{len(cajas)} cajas por etiquetar -> {salida}")

    print("bajando imagenes de la muestra...")
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda f: bajar(f, carpeta), cajas.file_name.unique()))

    etiquetar(cajas, carpeta, salida, mostrar_previa=not args.repetir)
    t = pd.read_csv(salida)
    print(t.groupby(["split", "nivel"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
