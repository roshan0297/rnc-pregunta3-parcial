"""EigenCAM sobre el backbone de YOLO (entregable 3).

EigenCAM (Muhammad & Yeasin, 2020) no usa gradientes: proyecta las activaciones
de una capa sobre su primera componente principal. Sirve bien para detectores,
donde no hay un unico logit de clase del que derivar.

Ademas de las figuras se calcula una medida cuantitativa por imagen:
    frac_en_caja = energia del mapa dentro de las cajas reales / energia total
    lift         = frac_en_caja / (area de las cajas / area de la imagen)
lift ~ 1 significa que el mapa no mira al animal mas que a cualquier otra
zona; lift >> 1 significa que se concentra en el animal.

Uso:
    python src/eigencam.py --pesos runs/E3_principal/weights/best.pt \
        --imgs data/raw/imagenes --out runs/E3_principal/eigencam
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

from cct import cargar
from tamano_oclusion import iou


class EigenCAM:
    def __init__(self, modelo_yolo, capa, imgsz=640):
        self.red = modelo_yolo.model.eval()
        self.red.to("cuda" if torch.cuda.is_available() else "cpu")
        self.imgsz = imgsz
        self.act = None
        self.red.model[capa].register_forward_hook(self._guardar)

    def _guardar(self, modulo, entrada, salida):
        self.act = (salida[0] if isinstance(salida, (list, tuple)) else salida).detach()

    @torch.no_grad()
    def __call__(self, img_pil):
        # se redimensiona sin letterbox: deforma un poco, pero el mapa vuelve al
        # tamano original con la misma deformacion inversa
        x = img_pil.convert("RGB").resize((self.imgsz, self.imgsz))
        # el predict de ultralytics puede mover o fusionar la red: se lee el dispositivo cada vez
        p = next(self.red.parameters())
        x = torch.from_numpy(np.array(x)).permute(2, 0, 1).to(p.device, p.dtype).div(255)[None]
        self.red(x)
        a = self.act[0].float()                       # (C, H, W)
        c, h, w = a.shape
        m = a.reshape(c, -1).T                        # (HW, C)
        m = m - m.mean(0)
        _, _, vh = torch.linalg.svd(m, full_matrices=False)
        cam = (m @ vh[0]).reshape(h, w)
        if cam.max() < -cam.min():                    # el signo de la SVD es arbitrario
            cam = -cam
        cam = cam.clamp(min=0)
        cam = F.interpolate(cam[None, None], size=(img_pil.height, img_pil.width),
                            mode="bilinear", align_corners=False)[0, 0]
        return (cam / (cam.max() + 1e-9)).cpu().numpy()


def medir(cam, cajas_px):
    mascara = np.zeros_like(cam, bool)
    for x1, y1, x2, y2 in cajas_px:
        mascara[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] = True
    frac = cam[mascara].sum() / (cam.sum() + 1e-9)
    area = mascara.mean()
    return frac, area, frac / max(area, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", required=True)
    ap.add_argument("--anot", default="data/raw/eccv_18_annotation_files")
    ap.add_argument("--imgs", default="data/raw/imagenes")
    ap.add_argument("--capa", type=int, default=9, help="indice de capa; 9 = SPPF, fin del backbone en YOLOv8")
    ap.add_argument("--n", type=int, default=150, help="imagenes por split")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--excluir-clases", default="car")
    ap.add_argument("--ejemplos", type=int, default=6, help="ejemplos por split en la figura")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    imagenes, cajas, _ = cargar(args.anot)
    malas = set(cajas[cajas.especie.isin(args.excluir_clases.split(","))].image_id)
    rutas = {p.name: p for p in Path(args.imgs).rglob("*.jpg")}
    sel = imagenes[imagenes.split.isin(["cis_test", "trans_test"]) & (imagenes.estado == "con_caja")
                   & ~imagenes.id.isin(malas) & imagenes.file_name.isin(rutas)]
    sel = pd.concat([g.sample(min(args.n, len(g)), random_state=args.semilla) for _, g in sel.groupby("split")])

    modelo = YOLO(args.pesos)
    cam_fn = EigenCAM(modelo, args.capa, args.imgsz)
    filas, mapas = [], {}
    for k, im in enumerate(sel.itertuples(), 1):
        img = Image.open(rutas[im.file_name]).convert("RGB")
        esc = img.width / im.width                    # las imagenes "sm" estan reducidas
        g = cajas[cajas.image_id == im.id]
        cajas_px = np.stack([g.x, g.y, g.x + g.w, g.y + g.h], 1) * esc
        cam = cam_fn(img)
        frac, area, lift = medir(cam, cajas_px)

        r = modelo.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        pred = r.boxes.xyxy.cpu().numpy()
        esp_pred = [modelo.names[int(c)] for c in r.boxes.cls.tolist()]
        correcta = any(e == esp and iou(caja, pred[[j]])[0] >= 0.5
                       for caja, esp in zip(cajas_px, g.especie) for j, e in enumerate(esp_pred))
        filas.append({"file_name": im.file_name, "split": im.split, "location": im.location,
                      "especie": g.especie.iloc[0], "frac_en_caja": frac, "area_cajas": area,
                      "lift": lift, "deteccion_correcta": correcta,
                      "pred": ",".join(esp_pred) or "-"})
        mapas[im.file_name] = (img, cam, cajas_px, list(zip(pred, esp_pred)))
        print(f"\r  {k}/{len(sel)}", end="")
    print()

    t = pd.DataFrame(filas)
    t.to_csv(out / "eigencam_por_imagen.csv", index=False)
    res = t.groupby(["split", "deteccion_correcta"]).agg(
        n=("lift", "size"), frac_en_caja=("frac_en_caja", "mean"),
        area_cajas=("area_cajas", "mean"), lift_mediana=("lift", "median"))
    res.to_csv(out / "eigencam_resumen.csv")
    print(res.round(3))

    # figura: en cada split, la mitad aciertos y la mitad fallos con el lift mas bajo
    fig, axes = plt.subplots(2, args.ejemplos, figsize=(3.2 * args.ejemplos, 6.2))
    for fila_ax, split in zip(axes, ["cis_test", "trans_test"]):
        s = t[t.split == split]
        mitad = args.ejemplos // 2
        elegidas = pd.concat([s[s.deteccion_correcta].sample(min(mitad, s.deteccion_correcta.sum()), random_state=0),
                              s[~s.deteccion_correcta].nsmallest(args.ejemplos - mitad, "lift")])
        for ax, fila in zip(fila_ax, elegidas.itertuples()):
            img, cam, cajas_px, preds = mapas[fila.file_name]
            ax.imshow(img)
            ax.imshow(cam, cmap="jet", alpha=0.45)
            for x1, y1, x2, y2 in cajas_px:
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="white", lw=1.5))
            for (x1, y1, x2, y2), e in preds:
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=1, ls="--"))
            ax.set_title(f"{split.replace('_test', '')} · cám. {fila.location}\n"
                         f"{fila.especie} → {fila.pred} | lift {fila.lift:.1f}",
                         fontsize=13, color="black" if fila.deteccion_correcta else "red")
        for ax in fila_ax:
            ax.axis("off")
    fig.suptitle("EigenCAM (capa %d). Blanco: caja real, rojo: predicción" % args.capa, fontsize=15)
    fig.tight_layout()
    fig.savefig(out / "eigencam_ejemplos.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
