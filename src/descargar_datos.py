"""Descarga de CCT-20 (LILA BC) y, opcionalmente, de imagenes vacias adicionales.

Uso:
    python src/descargar_datos.py                 # anotaciones + imagenes CCT-20 (~6 GB)
    python src/descargar_datos.py --solo-anotaciones
    python src/descargar_datos.py --vacias-extra 2000
"""
import argparse
import collections
import json
import random
import tarfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

BASE = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps"
URL_ANOT = f"{BASE}/eccv_18_annotations.tar.gz"
URL_IMGS = f"{BASE}/eccv_18_all_images_sm.tar.gz"
URL_FULL = f"{BASE}/labels/caltech_camera_traps.json.zip"
URL_IMG_FULL = "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/{}"

SPLITS = ["train", "cis_val", "trans_val", "cis_test", "trans_test"]
CAT_VACIO = 30


def descargar(url, destino):
    """Descarga con reanudacion: si el archivo existe a medias, pide solo lo que falta."""
    destino = Path(destino)
    ya = destino.stat().st_size if destino.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={ya}-"} if ya else {})
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # rango no satisfacible -> ya estaba completo
            print(f"  ya existe: {destino.name}")
            return destino
        raise
    total = ya + int(resp.headers.get("Content-Length", 0))
    modo = "ab" if resp.status == 206 else "wb"
    if modo == "wb":
        ya = 0
    with open(destino, modo) as f:
        while True:
            bloque = resp.read(1 << 20)
            if not bloque:
                break
            f.write(bloque)
            ya += len(bloque)
            print(f"\r  {destino.name}: {ya / 1e9:.2f} / {total / 1e9:.2f} GB", end="")
    print()
    return destino


def ubicaciones_cct20(dir_anot):
    locs = set()
    for s in SPLITS:
        d = json.load(open(dir_anot / f"{s}_annotations.json"))
        locs |= {str(i["location"]) for i in d["images"]}
    return locs


def descargar_vacias_extra(raw, n, semilla, lado_max=1024):
    """Vacias de ubicaciones que NO estan en CCT-20 (ni train ni test).

    El split train de CCT-20 no trae ninguna imagen vacia, asi que para ensenarle
    al detector que es "fondo" hay que traerlas de otro lado. Se reparten por
    ubicacion para que ninguna camara domine.
    """
    zip_full = descargar(URL_FULL, raw / "caltech_camera_traps.json.zip")
    z = zipfile.ZipFile(zip_full)
    full = json.loads(z.read(z.namelist()[0]))

    excluir = ubicaciones_cct20(raw / "eccv_18_annotation_files")
    cats = collections.defaultdict(set)
    for a in full["annotations"]:
        cats[a["image_id"]].add(a["category_id"])
    imgs = {i["id"]: i for i in full["images"]}

    por_loc = collections.defaultdict(list)
    for img_id, c in cats.items():
        im = imgs[img_id]
        if c == {CAT_VACIO} and str(im["location"]) not in excluir:
            por_loc[str(im["location"])].append(im)

    rng = random.Random(semilla)
    for lista in por_loc.values():
        rng.shuffle(lista)
    # round-robin entre ubicaciones hasta juntar n
    elegidas, i = [], 0
    while len(elegidas) < n and any(i < len(v) for v in por_loc.values()):
        for loc in sorted(por_loc):
            if i < len(por_loc[loc]) and len(elegidas) < n:
                elegidas.append(por_loc[loc][i])
        i += 1

    dir_out = raw / "vacias_extra"
    dir_out.mkdir(exist_ok=True)

    def bajar_una(im):
        out = dir_out / im["file_name"]
        if out.exists():
            return
        tmp = out.with_suffix(".tmp")
        urllib.request.urlretrieve(URL_IMG_FULL.format(im["file_name"]), tmp)
        # mismo tamano que las imagenes "sm" de CCT-20
        with Image.open(tmp) as img:
            img.thumbnail((lado_max, lado_max))
            img.save(out, quality=90)
        tmp.unlink()

    # son ~600 KB por imagen: en paralelo baja de ~20 min a ~2 min
    with ThreadPoolExecutor(max_workers=16) as pool:
        for k, _ in enumerate(pool.map(bajar_una, elegidas), 1):
            print(f"\r  vacias extra: {k}/{len(elegidas)}", end="")
    print()
    json.dump(elegidas, open(raw / "vacias_extra.json", "w"), indent=1)
    print(f"  {len(elegidas)} vacias de {len({e['location'] for e in elegidas})} ubicaciones")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--solo-anotaciones", action="store_true")
    ap.add_argument("--vacias-extra", type=int, default=0)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--borrar-tar", action="store_true", help="borra el .tar.gz tras descomprimir")
    args = ap.parse_args()

    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)

    tar_anot = descargar(URL_ANOT, raw / "eccv_18_annotations.tar.gz")
    if not (raw / "eccv_18_annotation_files").exists():
        tarfile.open(tar_anot).extractall(raw)

    if not args.solo_anotaciones:
        tar_imgs = descargar(URL_IMGS, raw / "eccv_18_all_images_sm.tar.gz")
        if not (raw / "imagenes_ok").exists():
            print("  descomprimiendo imagenes (tarda varios minutos)...")
            tarfile.open(tar_imgs).extractall(raw / "imagenes")
            (raw / "imagenes_ok").touch()
        if args.borrar_tar:  # en Kaggle el disco de trabajo es de ~20 GB
            tar_imgs.unlink()

    if args.vacias_extra:
        descargar_vacias_extra(raw, args.vacias_extra, args.semilla)


if __name__ == "__main__":
    main()
