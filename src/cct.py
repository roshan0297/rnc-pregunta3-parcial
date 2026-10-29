"""Lectura de las anotaciones de CCT-20 en una tabla por imagen y otra por caja."""
import json
from pathlib import Path

import pandas as pd

SPLITS = ["train", "cis_val", "trans_val", "cis_test", "trans_test"]
CAT_VACIO = 30


def cargar(dir_anot="data/raw/eccv_18_annotation_files"):
    """Devuelve (imagenes, cajas, categorias).

    imagenes: una fila por imagen con su split, ubicacion, fecha, especie(s) y
              un 'estado': 'vacia', 'con_caja' o 'sin_caja' (animal etiquetado
              a nivel de imagen pero sin bounding box).
    cajas:    una fila por bounding box, con area relativa ya calculada.
    """
    dir_anot = Path(dir_anot)
    filas_img, filas_caja, categorias = [], [], {}
    for split in SPLITS:
        d = json.load(open(dir_anot / f"{split}_annotations.json"))
        categorias.update({c["id"]: c["name"] for c in d["categories"]})
        anots = {}
        for a in d["annotations"]:
            anots.setdefault(a["image_id"], []).append(a)

        for im in d["images"]:
            lista = anots.get(im["id"], [])
            cats = sorted({a["category_id"] for a in lista})
            n_cajas = sum("bbox" in a for a in lista)
            if cats == [CAT_VACIO]:
                estado = "vacia"
            elif n_cajas > 0:
                estado = "con_caja"
            else:
                estado = "sin_caja"
            filas_img.append({
                "id": im["id"], "file_name": im["file_name"], "split": split,
                "location": str(im["location"]), "date_captured": im["date_captured"],
                "seq_id": im["seq_id"], "frame_num": im["frame_num"],
                "width": im["width"], "height": im["height"],
                "categorias": cats, "n_cajas": n_cajas, "estado": estado,
            })
            for a in lista:
                if "bbox" not in a:
                    continue
                x, y, w, h = a["bbox"]
                filas_caja.append({
                    "image_id": im["id"], "split": split, "category_id": a["category_id"],
                    "x": x, "y": y, "w": w, "h": h,
                    # el bbox esta en pixeles de la imagen ORIGINAL (width x height del json)
                    "area_rel": (w * h) / (im["width"] * im["height"]),
                })

    imagenes = pd.DataFrame(filas_img)
    imagenes["date_captured"] = pd.to_datetime(imagenes["date_captured"])
    cajas = pd.DataFrame(filas_caja)
    cajas["especie"] = cajas["category_id"].map(categorias)
    return imagenes, cajas, categorias
