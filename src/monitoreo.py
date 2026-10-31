"""Modulo de monitoreo ecologico basado en reglas (entregable 5).

Toma las detecciones por fotograma de UNA camara y las agrega como lo haria
un equipo de manejo de fauna:
  R1. Una deteccion cuenta si su confianza >= --conf.
  R2. Fotos de la misma camara separadas por <= --gap segundos forman un
      evento (la camara dispara rafagas; contar fotos infla la actividad).
      Se usa la hora de la foto y no el seq_id, porque en campo solo se
      tiene la hora.
  R3. Especie principal del evento = la de mayor suma de confianzas.
  R4. Individuos de una especie en el evento = maximo de detecciones en un
      mismo fotograma (no la suma: el mismo animal sale en varias fotos).
  R5. Evento sin detecciones = activacion vacia (viento, luz, vegetacion).
El mismo procedimiento se aplica a las etiquetas reales para comparar el
reporte del modelo con el que se obtendria revisando a mano.

Uso:
    python src/monitoreo.py --pred runs/E3_principal/predicciones.csv --ubicacion 46 \
        --out runs/E3_principal/monitoreo
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cct import cargar

NOCHE = set(range(19, 24)) | set(range(0, 6))
NOMBRES = {"opossum": "zarigüeya", "bobcat": "lince", "raccoon": "mapache", "coyote": "coyote",
           "rabbit": "conejo", "cat": "gato", "dog": "perro", "skunk": "zorrillo", "bird": "ave",
           "rodent": "roedor", "squirrel": "ardilla", "deer": "ciervo", "fox": "zorro", "badger": "tejón"}


def conteos_modelo(pred, conf):
    p = pred[(pred.especie.notna()) & (pred.especie != "") & (pred.conf >= conf)]
    cuenta = p.groupby(["file_name", "especie"]).size().rename("n").reset_index()
    voto = p.groupby(["file_name", "especie"]).conf.sum().rename("voto").reset_index()
    return cuenta.merge(voto, on=["file_name", "especie"])


def conteos_reales(imagenes, cajas, categorias, excluir):
    filas = []
    n_cajas = cajas.groupby(["image_id", "especie"]).size()
    for im in imagenes.itertuples():
        for cid in im.categorias:
            esp = categorias[cid]
            if esp == "empty" or esp in excluir:
                continue
            # imagenes sin caja: se sabe la especie pero no cuantos -> al menos 1
            n = int(n_cajas.get((im.id, esp), 0)) or 1
            filas.append({"file_name": im.file_name, "especie": esp, "n": n, "voto": float(n)})
    return pd.DataFrame(filas, columns=["file_name", "especie", "n", "voto"])


def agrupar_eventos(fotos, gap):
    fotos = fotos.sort_values("date_captured").reset_index(drop=True)
    salto = fotos.date_captured.diff().dt.total_seconds().fillna(gap + 1) > gap
    fotos["evento"] = salto.cumsum()
    return fotos


def resumir_eventos(fotos, conteos, sufijo):
    d = fotos[["file_name", "evento"]].merge(conteos, on="file_name")
    filas = []
    for ev, g in fotos.groupby("evento"):
        de = d[d.evento == ev]
        fila = {"evento": ev, "inicio": g.date_captured.min(), "fin": g.date_captured.max(), "n_fotos": len(g)}
        if de.empty:
            fila.update({f"especie_{sufijo}": "vacio", f"especies_{sufijo}": "", f"individuos_{sufijo}": 0})
        else:
            votos = de.groupby("especie").voto.sum().sort_values(ascending=False)
            indiv = de.groupby("especie").n.max()
            fila.update({f"especie_{sufijo}": votos.index[0],
                         f"especies_{sufijo}": ",".join(votos.index),
                         f"individuos_{sufijo}": int(indiv[votos.index[0]])})
        filas.append(fila)
    return pd.DataFrame(filas)


def reporte_especies(ev, sufijo):
    e = ev[ev[f"especie_{sufijo}"] != "vacio"]
    return e.groupby(f"especie_{sufijo}").agg(
        eventos=("evento", "size"),
        individuos_max=(f"individuos_{sufijo}", "max"),
        individuos_total=(f"individuos_{sufijo}", "sum"),
        pct_nocturno=("inicio", lambda s: s.dt.hour.isin(NOCHE).mean()),
        primera=("inicio", "min"), ultima=("inicio", "max"),
    ).rename_axis("especie")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="CSV de predecir.py")
    ap.add_argument("--anot", default="data/raw/eccv_18_annotation_files")
    ap.add_argument("--ubicacion", default=None, help="camara a reportar; sin valor lista las disponibles")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--gap", type=float, default=60, help="segundos entre fotos del mismo evento")
    ap.add_argument("--periodo", default="W", help="periodo de la tabla de actividad: D, W o M")
    ap.add_argument("--excluir-clases", default="car")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    imagenes, cajas, categorias = cargar(args.anot)
    pred = pd.read_csv(args.pred)
    imagenes = imagenes[imagenes.file_name.isin(pred.file_name)]
    if args.ubicacion is None:
        print("camaras disponibles (fotos por split):")
        print(imagenes.groupby(["location", "split"]).size().unstack(fill_value=0))
        return
    out = Path(args.out) / f"camara_{args.ubicacion}"
    out.mkdir(parents=True, exist_ok=True)

    fotos = agrupar_eventos(imagenes[imagenes.location == str(args.ubicacion)], args.gap)
    excluir = set(args.excluir_clases.split(","))
    ev_m = resumir_eventos(fotos, conteos_modelo(pred, args.conf), "modelo")
    ev_r = resumir_eventos(fotos, conteos_reales(fotos, cajas, categorias, excluir), "real")
    ev = ev_m.merge(ev_r.drop(columns=["inicio", "fin", "n_fotos"]), on="evento")
    ev.to_csv(out / "eventos.csv", index=False)

    # reporte por especie: modelo vs revision manual
    rep = reporte_especies(ev, "modelo").join(reporte_especies(ev, "real"), lsuffix="_modelo",
                                              rsuffix="_real", how="outer")
    rep.to_csv(out / "reporte_especies.csv")

    # actividad por periodo (lo que pediria un equipo de manejo)
    act = (ev[ev.especie_modelo != "vacio"]
           .groupby([pd.Grouper(key="inicio", freq=args.periodo), "especie_modelo"]).size()
           .unstack(fill_value=0))
    act.to_csv(out / f"actividad_por_periodo_{args.periodo}.csv")

    fotos_vacias_m = 1 - fotos.file_name.isin(conteos_modelo(pred, args.conf).file_name).mean()
    resumen = pd.Series({
        "camara": args.ubicacion,
        "fotos": len(fotos),
        "periodo": f"{fotos.date_captured.min():%Y-%m-%d} a {fotos.date_captured.max():%Y-%m-%d}",
        "eventos": len(ev),
        "pct_fotos_vacias_modelo": fotos_vacias_m,
        "pct_fotos_vacias_real": (fotos.estado == "vacia").mean(),
        "pct_eventos_vacios_modelo": (ev.especie_modelo == "vacio").mean(),
        "pct_eventos_vacios_real": (ev.especie_real == "vacio").mean(),
        "especies_modelo": ev.loc[ev.especie_modelo != "vacio", "especie_modelo"].nunique(),
        "especies_real": ev.loc[ev.especie_real != "vacio", "especie_real"].nunique(),
        "acierto_especie_por_evento": (ev.especie_modelo == ev.especie_real).mean(),
        # eventos con animal que el modelo marco como vacios: nadie los revisaria
        "eventos_con_animal_perdidos": int(((ev.especie_modelo == "vacio") & (ev.especie_real != "vacio")).sum()),
        "eventos_falsos_con_animal": int(((ev.especie_modelo != "vacio") & (ev.especie_real == "vacio")).sum()),
    })
    resumen.to_csv(out / "resumen.csv", header=False)
    print(resumen.to_string())
    print("\nreporte por especie:\n", rep[["eventos_modelo", "eventos_real", "individuos_max_modelo",
                                          "individuos_max_real", "pct_nocturno_modelo"]].round(2))
    confusion = pd.crosstab(ev.especie_real, ev.especie_modelo)
    confusion.to_csv(out / "confusion_eventos.csv")

    # actividad por hora del dia, modelo vs real, especies principales
    top = ev.loc[ev.especie_real != "vacio", "especie_real"].value_counts().index[:4]
    fig, axes = plt.subplots(1, len(top), figsize=(2.9 * len(top), 2.8), squeeze=False)
    for ax, esp in zip(axes[0], top):
        for col, color, et in [("especie_real", "black", "real"), ("especie_modelo", "tab:orange", "modelo")]:
            h = ev.loc[ev[col] == esp, "inicio"].dt.hour
            ax.hist(h, bins=24, range=(0, 24), histtype="step", lw=2, color=color, label=et)
        ax.set_title(NOMBRES.get(esp, esp), fontsize=12)
        ax.set_xlabel("hora del día", fontsize=11)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.tick_params(labelsize=10)
    axes[0, 0].set_ylabel("eventos", fontsize=11)
    axes[0, 0].legend(fontsize=10)
    fig.suptitle(f"Cámara {args.ubicacion}: actividad por hora (eventos)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "actividad_hora.png", dpi=150)
    print("\nresultados en", out)


if __name__ == "__main__":
    main()
