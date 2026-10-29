"""Prueba de borrado: ¿el detector sigue "viendo" la especie sin el animal? (entregable 3)

Para imagenes con un solo animal se tapa su caja (con margen) con el color
medio de la imagen y se vuelve a correr el detector. Si aun asi predice la
misma especie, la decision se apoya en el contexto/fondo de la camara y no
en el animal. Se separa:
    misma_especie_en_zona  -> "alucina" el animal en el mismo lugar
    misma_especie_fuera    -> lo predice en otra parte de la escena

Uso:
    python src/prueba_borrado.py --pesos runs/E3_principal/weights/best.pt \
        --imgs data/raw/imagenes --out runs/E3_principal/borrado
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO

from cct import cargar
from tamano_oclusion import iou


def borrar(img, caja, margen):
    x1, y1, x2, y2 = caja
    mx, my = (x2 - x1) * margen, (y2 - y1) * margen
    zona = (max(x1 - mx, 0), max(y1 - my, 0), min(x2 + mx, img.width), min(y2 + my, img.height))
    color = tuple(int(v) for v in np.asarray(img).reshape(-1, 3).mean(0))
    copia = img.copy()
    ImageDraw.Draw(copia).rectangle(zona, fill=color)
    return copia, np.array(zona)


def predecir(modelo, img, conf, imgsz):
    r = modelo.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy(), [modelo.names[int(c)] for c in r.boxes.cls.tolist()], r.boxes.conf.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", required=True)
    ap.add_argument("--anot", default="data/raw/eccv_18_annotation_files")
    ap.add_argument("--imgs", default="data/raw/imagenes")
    ap.add_argument("--n", type=int, default=300, help="imagenes por split")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--margen", type=float, default=0.15, help="margen extra alrededor de la caja")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--excluir-clases", default="car")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    imagenes, cajas, _ = cargar(args.anot)
    modelo = YOLO(args.pesos)
    clases = set(modelo.names.values())
    rutas = {p.name: p for p in Path(args.imgs).rglob("*.jpg")}
    una = cajas[cajas.image_id.map(cajas.image_id.value_counts()) == 1]
    una = una[una.split.isin(["cis_test", "trans_test"]) & una.especie.isin(clases)
              & una.file_name.isin(rutas)]
    una = una[una.image_id.isin(imagenes[imagenes.estado == "con_caja"].id)]
    una = pd.concat([g.sample(min(args.n, len(g)), random_state=args.semilla) for _, g in una.groupby("split")])

    filas, ejemplos = [], []
    for k, c in enumerate(una.itertuples(), 1):
        img = Image.open(rutas[c.file_name]).convert("RGB")
        esc = img.width / c.img_w
        caja = np.array([c.x, c.y, c.x + c.w, c.y + c.h]) * esc

        p0, e0, _ = predecir(modelo, img, args.conf, args.imgsz)
        original_ok = any(e == c.especie and iou(caja, p0[[j]])[0] >= 0.5 for j, e in enumerate(e0))

        tapada, zona = borrar(img, caja, args.margen)
        p1, e1, s1 = predecir(modelo, tapada, args.conf, args.imgsz)
        misma = [j for j, e in enumerate(e1) if e == c.especie]
        en_zona = any(iou(zona, p1[[j]])[0] >= 0.3 for j in misma)
        fila = {"file_name": c.file_name, "split": c.split, "especie": c.especie,
                "original_ok": original_ok,
                "misma_especie_en_zona": en_zona,
                "misma_especie_fuera": bool(misma) and not en_zona,
                "otra_especie": any(e != c.especie for e in e1),
                "conf_max_misma": max([s1[j] for j in misma], default=0.0)}
        filas.append(fila)
        if misma and len(ejemplos) < 8:
            ejemplos.append((tapada, zona, [(p1[j], e1[j], s1[j]) for j in misma], fila))
        print(f"\r  {k}/{len(una)}", end="")
    print()

    t = pd.DataFrame(filas)
    t["atajo"] = t.misma_especie_en_zona | t.misma_especie_fuera
    t.to_csv(out / "borrado_por_imagen.csv", index=False)
    cols = ["misma_especie_en_zona", "misma_especie_fuera", "atajo", "otra_especie"]
    res = pd.concat({
        "todas": t.groupby("split")[cols].mean(),
        "solo_detectadas_al_inicio": t[t.original_ok].groupby("split")[cols].mean(),
    }, names=["subconjunto"])
    res["n"] = pd.concat({"todas": t.groupby("split").size(),
                          "solo_detectadas_al_inicio": t[t.original_ok].groupby("split").size()})
    res.to_csv(out / "borrado_resumen.csv")
    print(res.round(3))
    por_esp = t.groupby(["split", "especie"]).atajo.agg(["mean", "size"]).unstack(0)
    por_esp.to_csv(out / "borrado_por_especie.csv")
    print("\npor especie:\n", por_esp.round(3))

    if ejemplos:
        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        for ax in axes.flat:
            ax.axis("off")
        for ax, (img, zona, preds, f) in zip(axes.flat, ejemplos):
            ax.imshow(img)
            ax.add_patch(patches.Rectangle(zona[:2], *(zona[2:] - zona[:2]), fill=False, ec="white", ls=":"))
            for (x1, y1, x2, y2), e, s in preds:
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=1.5))
                ax.text(x1, y1, f"{e} {s:.2f}", color="red", fontsize=7, va="bottom")
            ax.set_title(f"{f['split']}: {f['especie']} borrado", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "borrado_ejemplos.png", dpi=120)


if __name__ == "__main__":
    main()
