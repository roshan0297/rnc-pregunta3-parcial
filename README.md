# Pregunta 3 — Detección de fauna en cámaras trampa (YOLO, CCT-20)

Ajuste fino de YOLOv8 sobre Caltech Camera Traps (CCT-20), evaluación en cámaras vistas (`cis`) y nuevas (`trans`), análisis de desbalance, fondo, tamaño y oclusión, y un módulo de monitoreo ecológico. El informe está en `informe/` (LaTeX, plantilla NeurIPS).

## Ejecución con un solo comando

Requiere GPU NVIDIA, unos 20 GB de disco libres y conexión a internet (descarga ~7 GB desde LILA).

```bash
pip install -r requirements.txt
python run_all.py
```

`run_all.py` hace descarga → exploración → conversión → E1/E2 → E3 → evaluación cis/trans → umbral en validación → predicciones → todos los análisis. Cada paso se salta si su resultado ya existe, así que tras un corte basta con volver a lanzar el mismo comando: los entrenamientos se reanudan desde `last.pt` y los terminados no se repiten. Tiempo total en una T4: unas 6–8 h, de las que unas 5 son entrenamiento.

Opciones útiles:

| Opción | Para qué |
|---|---|
| `--data RUTA` | dónde guardar los datos (conviene un disco local rápido) |
| `--runs RUTA` | dónde guardar pesos y resultados |
| `--conf 0.5` | fija el umbral de confianza en vez de elegirlo con la curva F1 de validación |
| `--camara 46` | cámara de `trans_test` para el demo de monitoreo |
| `--rehacer-analisis` | repite EigenCAM, la prueba de borrado y el monitoreo |

### En Google Colab

Entorno de ejecución con GPU T4. Los resultados quedan en Drive para no perderlos si se corta la sesión:

```python
from google.colab import drive
drive.mount("/content/drive")
!git clone -q https://github.com/roshan0297/rnc-pregunta3-parcial.git /content/pregunta3
%cd /content/pregunta3
!pip install -q -r requirements.txt
!python run_all.py --data /content/data --runs /content/drive/MyDrive/pregunta3_runs --conf 0.5
```

El cuaderno `notebooks/entrenamiento_colab_kaggle.ipynb` hace lo mismo paso a paso, con salidas intermedias.

## Etiquetado de oclusión (manual, en una PC con pantalla)

CCT-20 no anota la oclusión. `src/etiquetar_oclusion.py` muestra 150 cajas por split de test y se etiquetan con las teclas `1` (bajo, <25 %), `2` (medio, 25–60 %), `3` (alto, >60 %), `s` (saltar) y `q` (salir y guardar):

```bash
python src/descargar_datos.py --solo-anotaciones
python src/etiquetar_oclusion.py               # 300 cajas
python src/etiquetar_oclusion.py --repetir 50  # re-etiquetado a ciegas para el kappa
```

Las etiquetas quedan en `results/oclusion/`. Al subirlas al repo, `run_all.py` las usa en el paso 10 (recall por oclusión y kappa de autoconsistencia).

## Scripts

| Script | Entregable | Qué produce |
|---|---|---|
| `src/descargar_datos.py` | — | Anotaciones, imágenes CCT-20 y vacías de otras ubicaciones (reanudable) |
| `src/explorar.py` | — | Especies, splits, ubicaciones, tamaño de cajas, hora de captura |
| `src/convertir_yolo.py` | 1 | Dataset YOLO con exclusiones documentadas y fracción de vacías configurable |
| `run_all.py` (pasos 4–7) | 1, 2 | E1/E2/E3, comparación de vacías, mAP por especie cis/trans, falsos disparos |
| `src/predecir.py` | — | Predicciones de todo `cis_test` y `trans_test` en un CSV (las usan los demás) |
| `src/analisis_desbalance.py` | 2 | AP por especie vs nº de cajas de entrenamiento |
| `src/eigencam.py` | 3 | EigenCAM propio sobre el backbone y medida de concentración en el animal (lift) |
| `src/prueba_borrado.py` | 3 | % de imágenes donde se sigue prediciendo la especie tras tapar al animal |
| `src/etiquetar_oclusion.py` | 4 | Herramienta local para etiquetar oclusión (bajo/medio/alto) |
| `src/tamano_oclusion.py` | 4 | Recall por tamaño de caja y por nivel de oclusión, cis vs trans, kappa |
| `src/monitoreo.py` | 5 | Reporte por cámara: eventos, especies, individuos, actividad por hora/periodo, modelo vs real |

## Resultados principales (modelo E3, YOLOv8s 640)

| | mAP@0.5 | mAP@0.5:0.95 | Falsos disparos en vacías |
|---|---|---|---|
| `cis_test` | 0.721 | 0.463 | 3.6 % |
| `trans_test` | 0.459 | 0.286 | 4.3 % |

Detalle por especie, análisis y discusión en el informe (`informe/main.pdf`).
