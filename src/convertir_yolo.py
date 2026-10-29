"""Convierte CCT-20 (COCO Camera Traps) al formato de Ultralytics YOLO.

Decisiones (todas documentadas en data/yolo/metadatos.csv, columna 'motivo'):
  * Imagenes con animal pero SIN bounding box se excluyen de los splits de
    deteccion: si entraran con un .txt vacio, YOLO aprenderia que ese animal
    es fondo, y en test contarian como falsos positivos.
  * Clases en --excluir-clases (por defecto 'car', cuyas cajas son la imagen
    completa) se sacan junto con sus imagenes.
  * Cajas que se salen de la imagen se recortan al borde.
  * Vacias: train no trae ninguna, se agregan de data/raw/vacias_extra si
    --frac-vacias > 0. En val/test se dejan las vacias reales (con .txt vacio).

Uso:
    python src/convertir_yolo.py --imgs data/raw/imagenes --frac-vacias 0.2
    python src/convertir_yolo.py --imgs data/raw/imagenes --limite 200   # prueba rapida
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from cct import SPLITS, cargar


def enlazar(src, dst):
    """Hardlink si se puede (no duplica los 6 GB), si no copia."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def linea_yolo(c, w_img, h_img, clase):
    x1, y1 = max(c.x, 0), max(c.y, 0)
    x2, y2 = min(c.x + c.w, w_img), min(c.y + c.h, h_img)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    xc, yc = (x1 + x2) / 2 / w_img, (y1 + y2) / 2 / h_img
    return f"{clase} {xc:.6f} {yc:.6f} {(x2 - x1) / w_img:.6f} {(y2 - y1) / h_img:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anot", default="data/raw/eccv_18_annotation_files")
    ap.add_argument("--imgs", default="data/raw/imagenes")
    ap.add_argument("--vacias-extra", default="data/raw/vacias_extra")
    ap.add_argument("--salida", default="data/yolo")
    ap.add_argument("--excluir-clases", default="car", help="separadas por coma; '' para ninguna")
    ap.add_argument("--min-cajas", type=int, default=0,
                    help="clases con menos cajas en train se excluyen (0 = mantener todas)")
    ap.add_argument("--frac-vacias", type=float, default=0.0,
                    help="vacias extra en train, como fraccion de las imagenes con animal")
    ap.add_argument("--limite", type=int, default=0, help="max imagenes por split (prueba)")
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()

    imagenes, cajas, categorias = cargar(args.anot)
    salida = Path(args.salida)

    # --- clases que se quedan ---
    excluir = {n for n in args.excluir_clases.split(",") if n}
    animales = [n for cid, n in categorias.items() if cid != 30]
    n_train = cajas[cajas.split == "train"].especie.value_counts().reindex(animales, fill_value=0)
    raras = set(n_train[n_train < args.min_cajas].index)
    fuera = excluir | raras
    clases = sorted(n for cid, n in categorias.items() if cid != 30 and n not in fuera)
    idx = {n: i for i, n in enumerate(clases)}
    print("clases:", clases)
    print("excluidas:", sorted(fuera) or "ninguna")

    # --- motivo de exclusion por imagen ---
    nombres = imagenes.categorias.apply(lambda cs: {categorias[c] for c in cs})
    imagenes["motivo"] = ""
    imagenes.loc[imagenes.estado == "sin_caja", "motivo"] = "animal_sin_caja"
    imagenes.loc[nombres.apply(lambda s: bool(s & fuera)), "motivo"] = "clase_excluida"
    imagenes["incluida"] = imagenes.motivo == ""

    if args.limite:
        imagenes = pd.concat([g.sample(min(len(g), args.limite), random_state=args.semilla)
                              for _, g in imagenes.groupby("split")])

    rutas = {p.name: p for p in Path(args.imgs).rglob("*.jpg")}
    faltan = ~imagenes.file_name.isin(rutas)
    imagenes.loc[faltan & imagenes.incluida, "motivo"] = "archivo_no_encontrado"
    imagenes.loc[faltan, "incluida"] = False

    cajas_por_img = {k: g for k, g in cajas[cajas.especie.isin(idx)].groupby("image_id")}
    for split in SPLITS:
        (salida / "images" / split).mkdir(parents=True, exist_ok=True)
        (salida / "labels" / split).mkdir(parents=True, exist_ok=True)
        sel = imagenes[(imagenes.split == split) & imagenes.incluida]
        for fila in sel.itertuples():
            enlazar(rutas[fila.file_name], salida / "images" / split / fila.file_name)
            lineas = []
            for c in cajas_por_img.get(fila.id, pd.DataFrame()).itertuples():
                l = linea_yolo(c, fila.width, fila.height, idx[c.especie])
                if l:
                    lineas.append(l)
            (salida / "labels" / split / fila.file_name).with_suffix(".txt").write_text("\n".join(lineas))
        print(f"{split:10s} {len(sel):6d} imagenes")

    # --- vacias extra para train ---
    extra_meta = Path(args.vacias_extra).with_suffix(".json")
    if args.frac_vacias > 0:
        if not extra_meta.exists():
            raise SystemExit("No hay vacias extra: corre descargar_datos.py --vacias-extra N")
        disponibles = json.load(open(extra_meta))
        n_animal = ((imagenes.split == "train") & imagenes.incluida).sum()
        n = min(int(args.frac_vacias * n_animal), len(disponibles))
        filas = []
        for im in pd.DataFrame(disponibles).sample(n, random_state=args.semilla).itertuples():
            src = Path(args.vacias_extra) / im.file_name
            enlazar(src, salida / "images" / "train" / im.file_name)
            (salida / "labels" / "train" / im.file_name).with_suffix(".txt").write_text("")
            filas.append({"id": im.id, "file_name": im.file_name, "split": "train",
                          "location": str(im.location), "estado": "vacia_extra",
                          "incluida": True, "motivo": ""})
        imagenes = pd.concat([imagenes, pd.DataFrame(filas)], ignore_index=True)
        print(f"train      +{n} vacias extra ({n / n_animal:.0%} de las imagenes con animal)")

    imagenes.to_csv(salida / "metadatos.csv", index=False)
    resumen = imagenes.groupby(["split", "motivo"]).size().unstack(fill_value=0)
    print("\nimagenes por split y motivo ('' = incluida):\n", resumen)

    # --- yaml: val = cis_val + trans_val; un yaml por cada test ---
    raiz = salida.resolve().as_posix()
    nombres_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(clases))
    for nombre, test in [("cct.yaml", "cis_test"), ("cct_trans.yaml", "trans_test")]:
        (salida / nombre).write_text(
            f"path: {raiz}\n"
            f"train: images/train\n"
            f"val: [images/cis_val, images/trans_val]\n"
            f"test: images/{test}\n"
            f"names:\n{nombres_yaml}\n")
    json.dump(vars(args), open(salida / "config_conversion.json", "w"), indent=1)


if __name__ == "__main__":
    main()
