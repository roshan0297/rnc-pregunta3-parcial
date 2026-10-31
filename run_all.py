"""Pregunta 3 de punta a punta con un solo comando.

    python run_all.py                                   # local: data/ y runs/ en el repo
    python run_all.py --data /content/data \\
        --runs /content/drive/MyDrive/pregunta3_runs    # Colab, resultados en Drive

Cada paso se salta si su resultado ya existe: si la sesion se corta, se vuelve a
lanzar el mismo comando. La descarga continua donde quedo, un entrenamiento a
medias se reanuda desde last.pt y uno terminado no se repite.

Pasos: descarga -> exploracion -> conversion (sin vacias / +20 % vacias) ->
E1, E2 y comparacion -> E3 -> evaluacion cis/trans -> umbral en validacion ->
predicciones -> desbalance, tamano/oclusion, EigenCAM, borrado, monitoreo.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent

COMUN = dict(
    imgsz=640, epochs=30, batch=32, seed=42, patience=10, exist_ok=True,
    fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.4, hsv_v=0.4,
    mosaic=1.0, close_mosaic=5,
)


def paso(titulo):
    print(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}", flush=True)


def correr(script, *args):
    subprocess.run([sys.executable, str(RAIZ / "src" / script), *map(str, args)], check=True, cwd=RAIZ)


def terminado(pesos):
    # al terminar (o por early stopping) ultralytics deja epoch = -1 en el checkpoint;
    # reanudar uno terminado no falla en v8.4: entrena coco8 desde cero, hay que evitarlo
    last = pesos / "last.pt"
    return last.exists() and torch.load(last, map_location="cpu", weights_only=False).get("epoch") == -1


def entrenar(runs, nombre, yaml, modelo, workers, **cambios):
    pesos = runs / nombre / "weights"
    if terminado(pesos):
        print(f"{nombre}: ya entrenado, se usa {pesos / 'best.pt'}")
    elif (pesos / "last.pt").exists():
        print(f"{nombre}: reanudando desde last.pt")
        YOLO(pesos / "last.pt").train(resume=True)
    else:
        YOLO(modelo).train(data=str(yaml), name=nombre, project=str(runs), workers=workers,
                           device=0 if torch.cuda.is_available() else "cpu", **{**COMUN, **cambios})
    return pesos / "best.pt"


def tasa_falsos_disparos(pesos, dir_yolo, splits, conf=0.5, imgsz=640):
    # % de imagenes vacias con al menos una deteccion de confianza >= conf
    modelo = YOLO(pesos)
    filas = []
    for s in splits:
        vacias = [dir_yolo / "images" / s / (t.stem + ".jpg")
                  for t in (dir_yolo / "labels" / s).glob("*.txt") if t.stat().st_size == 0]
        disparos = 0
        for i in range(0, len(vacias), 32):   # en lotes: una lista grande llena la GPU
            res = modelo.predict(vacias[i:i + 32], conf=conf, imgsz=imgsz, verbose=False)
            disparos += sum(len(r.boxes) > 0 for r in res)
        filas.append({"split": s, "n_vacias": len(vacias), "con_disparo": disparos,
                      "tasa": disparos / max(len(vacias), 1)})
    return pd.DataFrame(filas)


def evaluar(best, dir_yolo, imgsz, out):
    tablas = []
    for yaml, test in [("cct.yaml", "cis_test"), ("cct_trans.yaml", "trans_test")]:
        m = YOLO(best).val(data=str(dir_yolo / yaml), split="test", imgsz=imgsz,
                           project=str(out), name=f"eval_{test}", exist_ok=True)
        t = pd.DataFrame({"especie": [m.names[i] for i in m.box.ap_class_index],
                          "AP50": m.box.ap50, "AP50-95": m.box.ap})
        t.loc[len(t)] = ["TODAS (mAP)", m.box.map50, m.box.map]
        t.insert(0, "test", test)
        tablas.append(t)
    pd.concat(tablas).to_csv(out / "eval_por_especie.csv", index=False)
    tasa_falsos_disparos(best, dir_yolo, ("cis_test", "trans_test"), imgsz=imgsz) \
        .to_csv(out / "falsos_disparos_test.csv", index=False)


def umbral_validacion(best, dir_yolo, imgsz, out):
    # confianza que maximiza el F1 promedio de todas las clases en VALIDACION (nunca en test)
    archivo = out / "conf_elegido.txt"
    if archivo.exists():
        return float(archivo.read_text())
    m = YOLO(best).val(data=str(dir_yolo / "cct.yaml"), split="val", imgsz=imgsz, plots=False, verbose=False)
    f1 = m.box.f1_curve.mean(0)
    conf = round(float(m.box.px[f1.argmax()]), 2)
    archivo.write_text(str(conf))
    print(f"umbral elegido en validacion: {conf} (F1 medio {f1.max():.3f})")
    return conf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(RAIZ / "data"), help="datos crudos y convertidos (disco local rapido)")
    ap.add_argument("--runs", default=str(RAIZ / "runs"), help="pesos y resultados (en Colab, una carpeta de Drive)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--camara", default="46", help="camara de trans_test para el demo de monitoreo")
    ap.add_argument("--conf", type=float, default=None, help="fija el umbral en vez de elegirlo en validacion")
    ap.add_argument("--rehacer-analisis", action="store_true", help="repite EigenCAM, borrado y monitoreo")
    args = ap.parse_args()

    data, runs = Path(args.data), Path(args.runs)
    raw = data / "raw"
    anot, imgs = raw / "eccv_18_annotation_files", raw / "imagenes"
    runs.mkdir(parents=True, exist_ok=True)

    paso("1. Descarga (anotaciones, imagenes CCT-20 y 3000 vacias de otras ubicaciones)")
    correr("descargar_datos.py", "--raw", raw, "--vacias-extra", 3000, "--borrar-tar")

    paso("2. Exploracion")
    if not (runs / "exploracion").exists():
        correr("explorar.py", "--anot", anot, "--imgs", imgs, "--out", runs / "exploracion")

    paso("3. Conversion a YOLO")
    yolo = {"sin_vacias": data / "yolo_sin_vacias", "vacias20": data / "yolo_vacias20"}
    for frac, d in [(0, yolo["sin_vacias"]), (0.2, yolo["vacias20"])]:
        if not (d / "cct.yaml").exists():
            correr("convertir_yolo.py", "--anot", anot, "--imgs", imgs, "--vacias-extra", raw / "vacias_extra",
                   "--salida", d, "--frac-vacias", frac)

    paso("4. E1 y E2: efecto de las imagenes vacias (YOLOv8n)")
    best = {"sin_vacias": entrenar(runs, "E1_n640_sin_vacias", yolo["sin_vacias"] / "cct.yaml", "yolov8n.pt", args.workers),
            "vacias20": entrenar(runs, "E2_n640_vacias20", yolo["vacias20"] / "cct.yaml", "yolov8n.pt", args.workers)}
    comp_csv = runs / "comparacion_E1_E2.csv"
    if not comp_csv.exists():
        filas = []
        for exp, clave in [("E1", "sin_vacias"), ("E2", "vacias20")]:
            m = YOLO(best[clave]).val(data=str(yolo[clave] / "cct.yaml"), split="val", verbose=False, plots=False)
            fd = tasa_falsos_disparos(best[clave], yolo[clave], ("cis_val", "trans_val"))
            filas.append({"exp": exp, "dataset": clave, "mAP50": m.box.map50, "mAP50-95": m.box.map,
                          **{f"falsos_{r.split}": r.tasa for r in fd.itertuples()}})
        pd.DataFrame(filas).to_csv(comp_csv, index=False)
    comp = pd.read_csv(comp_csv)
    print(comp.round(3).to_string(index=False))

    paso("5. E3: modelo principal (YOLOv8s, dataset ganador de E1/E2)")
    if "dataset" not in comp:   # comparacion hecha desde el cuaderno, sin la columna
        comp["dataset"] = ["sin_vacias", "vacias20"]
    dataset = comp.loc[comp["mAP50-95"].idxmax(), "dataset"]
    print("dataset elegido:", dataset)
    best_e3 = entrenar(runs, "E3_principal", yolo[dataset] / "cct.yaml", "yolov8s.pt", args.workers,
                       epochs=50, batch=16)
    out = runs / "E3_principal"

    paso("6. Evaluacion en cis_test y trans_test")
    if not (out / "eval_por_especie.csv").exists():
        evaluar(best_e3, yolo[dataset], 640, out)
    ev = pd.read_csv(out / "eval_por_especie.csv")
    print(ev.pivot(index="especie", columns="test", values="AP50").round(3))

    paso("7. Umbral de confianza (validacion)")
    conf = args.conf if args.conf is not None else umbral_validacion(best_e3, yolo[dataset], 640, out)
    print("conf =", conf)

    paso("8. Predicciones sobre todo el test")
    pred = out / "predicciones.csv"
    if not pred.exists():
        correr("predecir.py", "--pesos", best_e3, "--anot", anot, "--imgs", imgs, "--salida", pred)

    paso("9. Entregable 2: AP por especie vs cajas de entrenamiento")
    correr("analisis_desbalance.py", "--eval", out / "eval_por_especie.csv", "--yolo", yolo[dataset],
           "--out", out / "desbalance")

    paso("10. Entregable 4: recall por tamano y oclusion")
    correr("tamano_oclusion.py", "--pred", pred, "--anot", anot, "--conf", conf,
           "--oclusion", RAIZ / "results" / "oclusion" / "etiquetas_oclusion.csv", "--out", out / "tamano_oclusion")

    paso("10b. Verificaciones: Spearman, concentracion por camara, cajas del borde, acuerdo de oclusion")
    correr("verificaciones.py", "--anot", anot, "--pred", pred, "--cajas", out / "tamano_oclusion" / "cajas_emparejadas.csv",
           "--ap", out / "desbalance" / "ap_vs_cajas_train.csv", "--oclusion", RAIZ / "results" / "oclusion",
           "--out", out / "verificaciones")

    paso("11. Entregable 3: EigenCAM y prueba de borrado")
    if args.rehacer_analisis or not (out / "eigencam" / "eigencam_resumen.csv").exists():
        correr("eigencam.py", "--pesos", best_e3, "--anot", anot, "--imgs", imgs, "--conf", conf,
               "--out", out / "eigencam")
    if args.rehacer_analisis or not (out / "borrado" / "borrado_resumen.csv").exists():
        correr("prueba_borrado.py", "--pesos", best_e3, "--anot", anot, "--imgs", imgs, "--conf", conf,
               "--out", out / "borrado")

    paso(f"12. Entregable 5: monitoreo de la camara {args.camara}")
    if args.rehacer_analisis or not (out / "monitoreo" / f"camara_{args.camara}" / "resumen.csv").exists():
        correr("monitoreo.py", "--pred", pred, "--anot", anot, "--conf", conf,
               "--ubicacion", args.camara, "--out", out / "monitoreo")

    print(f"\nListo. Resultados en {runs}")


if __name__ == "__main__":
    main()
