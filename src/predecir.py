"""Corre el detector sobre todas las imagenes de uno o varios splits y guarda un CSV.

Se predice una sola vez (con confianza baja) y los analisis posteriores aplican
su propio umbral: tamano/oclusion, monitoreo, etc. leen este CSV.

Uso:
    python src/predecir.py --pesos runs/E3_principal/weights/best.pt \
        --imgs data/raw/imagenes --splits cis_test trans_test --salida runs/E3_principal/predicciones.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from cct import cargar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", required=True)
    ap.add_argument("--anot", default="data/raw/eccv_18_annotation_files")
    ap.add_argument("--imgs", default="data/raw/imagenes")
    ap.add_argument("--splits", nargs="+", default=["cis_test", "trans_test"])
    ap.add_argument("--salida", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--lote", type=int, default=32)
    args = ap.parse_args()

    imagenes, _, _ = cargar(args.anot)
    rutas = {p.name: p for p in Path(args.imgs).rglob("*.jpg")}
    sel = imagenes[imagenes.split.isin(args.splits) & imagenes.file_name.isin(rutas)]
    lista = [rutas[f] for f in sel.file_name]
    print(f"{len(lista)} imagenes de {args.splits}")

    modelo = YOLO(args.pesos)
    filas = []
    # en lotes: una lista grande de rutas se procesa como un solo lote y llena la GPU
    for i in range(0, len(lista), args.lote):
        for r in modelo.predict(lista[i:i + args.lote], conf=args.conf, imgsz=args.imgsz, verbose=False):
            nombre = Path(r.path).name
            if len(r.boxes) == 0:
                filas.append({"file_name": nombre, "especie": "", "conf": 0.0})
            for cls, conf, (x1, y1, x2, y2) in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                                   r.boxes.xyxyn.tolist()):
                filas.append({"file_name": nombre, "especie": modelo.names[int(cls)], "conf": conf,
                              "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        print(f"\r  {min(i + args.lote, len(lista))}/{len(lista)}", end="")
    print()

    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(args.salida, index=False)
    print("guardado en", args.salida)


if __name__ == "__main__":
    main()
