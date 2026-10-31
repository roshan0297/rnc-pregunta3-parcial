"""Tasa de falsos disparos de un modelo en las fotos vacias de validacion (Tabla 1, fila E3).

Baja de LILA solo las vacias de cis_val y trans_val (unas 1 000, ~150 MB, resolucion
original) y cuenta en cuantas el modelo detecta algo con confianza >= --conf.
Corre en CPU en pocos minutos.

Uso:
    python src/falsos_disparos_val.py --pesos runs/E3_principal/weights/best.pt
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from cct import cargar
from etiquetar_oclusion import bajar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", required=True)
    ap.add_argument("--anot", default="data/raw/eccv_18_annotation_files")
    ap.add_argument("--imgs", default="data/vacias_val")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--salida", default="results/evaluacion/falsos_disparos_val_E3.csv")
    args = ap.parse_args()

    imagenes, _, _ = cargar(args.anot)
    v = imagenes[imagenes.split.isin(["cis_val", "trans_val"]) & (imagenes.estado == "vacia")]
    carpeta = Path(args.imgs)
    carpeta.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(32) as pool:
        rutas = list(pool.map(lambda f: bajar(f, carpeta), v.file_name))
    v = v.assign(ruta=rutas).dropna(subset=["ruta"])

    modelo = YOLO(args.pesos)
    filas = []
    for split, g in v.groupby("split"):
        r = [str(x) for x in g.ruta]
        disparos = 0
        for i in range(0, len(r), 32):
            res = modelo.predict(r[i:i + 32], conf=args.conf, imgsz=640, verbose=False)
            disparos += sum(len(x.boxes) > 0 for x in res)
        filas.append({"split": split, "n_vacias": len(r), "con_disparo": disparos, "tasa": disparos / len(r)})
    t = pd.DataFrame(filas)
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.salida, index=False)
    print(t)


if __name__ == "__main__":
    main()
