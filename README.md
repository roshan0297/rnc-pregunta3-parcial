# Pregunta 3 — Detección de fauna en cámaras trampa (YOLO, CCT-20)

Ajuste fino de YOLOv8 sobre Caltech Camera Traps (CCT-20), evaluación en cámaras vistas (`cis`) y nuevas (`trans`), análisis de desbalance, fondo, tamaño y oclusión, y un módulo de monitoreo ecológico.

- **Informe:** `informe/main.pdf` (fuente LaTeX en `informe/`; `informe_overleaf.zip` se sube tal cual a Overleaf).
- **Presentación:** `presentacion/presentacion_pregunta3_grupo6.pptx`, con el guion en `presentacion/speech_pregunta3.md`.

## Estructura

```
run_all.py              todo el flujo con un solo comando
src/                    un script por paso (ver tabla al final)
notebooks/              el mismo flujo paso a paso para Colab / Kaggle
results/                los CSV de cada análisis: las cifras del informe salen de aquí
informe/                informe NeurIPS (LaTeX + PDF + figuras)
presentacion/           diapositivas y guion
```

`data/` y `runs/` (datos, pesos y salidas pesadas) no se suben al repositorio.

---

## Verificar las cifras del informe

Hay tres niveles, de menor a mayor costo.

### Nivel 1 · Leer los CSV (sin instalar nada)

Cada número del informe está en un archivo de `results/`:

| Informe | Cifra | Archivo |
|---|---|---|
| Tabla 1 | mAP y falsos disparos de E1 y E2 en validación | `results/entrenamiento/comparacion_E1_E2.csv` |
| Tabla 1 | Falsos disparos de E3 en validación (5,1 % / 3,3 %) | `results/evaluacion/falsos_disparos_val_E3.csv` |
| Tabla 2, Fig. 1 | AP@0.5 y AP@0.5:0.95 por especie, cis y trans; mAP 0,721 / 0,459 | `results/evaluacion/eval_por_especie.csv` |
| Sec. 5.3 | Falsos disparos en test (3,6 % / 4,3 %) | `results/evaluacion/falsos_disparos_test.csv` |
| Sec. 5.2 | AP frente a cajas de train; Spearman 0,68 (p = 0,008) y 0,49 (p = 0,09) | `results/desbalance/ap_vs_cajas_train.csv`, `results/verificaciones/spearman_ap_vs_cajas.csv` |
| Sec. 5.3 | Concentración por cámara (gato 74 % en la cámara 38; ρ = 0,35, p = 0,30) | `results/verificaciones/concentracion_por_camara.csv` |
| Sec. 5.3 | Prueba de borrado (0/600 en la zona, 3/600 fuera) | `results/borrado/borrado_resumen.csv`, `borrado_por_especie.csv` |
| Tabla 3 | EigenCAM: energía en cajas y lift | `results/eigencam/eigencam_resumen.csv` |
| Tabla 4 | Recall por tamaño de caja | `results/tamano_oclusion/recall_por_grupo.csv` |
| Sec. 5.4 | Cajas grandes de trans: 80 % tocan el borde; mapache → coyote (251) | `results/verificaciones/cajas_grandes_trans.csv`, `confusiones_cajas_grandes_trans.csv` |
| Tabla 5 | Recall por nivel de oclusión | `results/tamano_oclusion/recall_por_grupo.csv` |
| Sec. 5.4 | Etiquetas de oclusión y acuerdo humano (κ = −0,21; 0,23 leído como visibilidad) | `results/oclusion/*.csv`, `results/verificaciones/acuerdo_oclusion*.csv` |
| Tabla 6, Fig. 3 | Monitoreo de la cámara 46 (eventos, especies, nocturnidad, 221 eventos perdidos) | `results/monitoreo/resumen.csv`, `reporte_especies.csv`, `eventos.csv` |
| Sec. 5.5 | Los 55 "gatos" (24 linces, 16 coyotes, 8 mapaches) | `results/monitoreo/confusion_eventos.csv` |
| Sec. 3 | Especies, splits, ubicaciones y tamaños | `results/exploracion/*.csv` |

### Nivel 2 · Recalcular los análisis sin GPU (~10 min)

Se necesitan las anotaciones (25 MB), las predicciones del modelo final y, para EigenCAM y el borrado, el modelo `best.pt`. Las predicciones y el modelo se generan con el nivel 3. Si se reciben ya hechos, se copian a `runs/E3_principal/`.

```bash
pip install -r requirements.txt
python src/descargar_datos.py --solo-anotaciones
```

Con `runs/E3_principal/predicciones.csv` disponible:

```bash
# Tablas 4 y 5, y kappa de oclusión
python src/tamano_oclusion.py --pred runs/E3_principal/predicciones.csv --conf 0.5 --out runs/E3_principal/tamano_oclusion
# Spearman, concentración por cámara, cajas del borde y acuerdo de oclusión
python src/verificaciones.py --pred runs/E3_principal/predicciones.csv --cajas runs/E3_principal/tamano_oclusion/cajas_emparejadas.csv
# Monitoreo de la cámara 46 (Tabla 6, Fig. 3)
python src/monitoreo.py --pred runs/E3_principal/predicciones.csv --conf 0.5 --ubicacion 46 --out runs/E3_principal/monitoreo
```

Con `runs/E3_principal/weights/best.pt` disponible (bajan de LILA solo las imágenes que necesitan):

```bash
# Tabla 1, falsos disparos de E3 en validación (~1 000 imágenes vacías)
python src/falsos_disparos_val.py --pesos runs/E3_principal/weights/best.pt
```

### Nivel 3 · Reproducir todo desde cero (GPU, ~6–8 h en una T4)

```bash
pip install -r requirements.txt
python run_all.py --conf 0.5
```

`run_all.py` hace descarga → exploración → conversión → E1/E2 → E3 → evaluación cis/trans → predicciones → todos los análisis. Cada paso se salta si su resultado ya existe; si la sesión se corta, basta con relanzar el mismo comando (los entrenamientos se reanudan desde `last.pt` y los terminados no se repiten). `--conf 0.5` fija el umbral usado en el informe; sin esa opción se elige en validación con la curva F1 (máximo en 0,45).

En Google Colab, con los resultados en Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
!git clone -q https://github.com/roshan0297/rnc-pregunta3-parcial.git /content/pregunta3
%cd /content/pregunta3
!pip install -q -r requirements.txt
!python run_all.py --data /content/data --runs /content/drive/MyDrive/pregunta3_runs --conf 0.5
```

| Opción de `run_all.py` | Para qué |
|---|---|
| `--data RUTA` / `--runs RUTA` | dónde guardar datos y resultados |
| `--conf 0.5` | fija el umbral de confianza |
| `--camara 46` | cámara de `trans_test` para el monitoreo |
| `--rehacer-analisis` | repite EigenCAM, borrado y monitoreo |

---

## Notas para quien verifique

- **Entorno usado:** Ultralytics 8.4, PyTorch 2.11, Tesla T4 en Colab, semilla 42. Con otra versión o GPU, el mAP puede variar en la tercera cifra decimal.
- **E3 se detuvo por early stopping** en la época 44 (mejor época: 34). El reentrenamiento parte de los mismos hiperparámetros, pero no garantiza los mismos pesos bit a bit.
- **Falsos disparos de E3 en validación:** se midieron sobre las imágenes a resolución original, mientras que E1 y E2 usaron la versión reducida de 1024 px. Por eso la Tabla 1 los marca como no estrictamente comparables.
- **Etiquetado de oclusión:** las 300 etiquetas de `results/oclusion/etiquetas_oclusion.csv` las asignó un modelo de lenguaje con visión siguiendo el criterio del informe. `etiquetas_oclusion_repeticion.csv` es el re-etiquetado a ciegas de 50 cajas por una persona. El acuerdo salió en espejo (κ = −0,21), consistente con una escala leída al revés; por eso el informe trata ese análisis como exploratorio. `desacuerdos_bajo_vs_alto.jpg` muestra los casos en conflicto. La herramienta (`src/etiquetar_oclusion.py`) ya tiene la leyenda corregida.
- **EigenCAM (Fig. 2):** la figura se regeneró localmente con `best.pt` sobre las mismas 12 imágenes del análisis de Colab, a resolución original. Por eso algún lift de la figura difiere en la primera decimal del de Colab; la Tabla 3 es la del análisis completo.

## Etiquetar la oclusión a mano

```bash
python src/etiquetar_oclusion.py               # 150 cajas por split de test
python src/etiquetar_oclusion.py --repetir 50  # re-etiquetado a ciegas para el kappa
```

En la ventana: **1** = casi nada tapado (<25 %), **2** = la mitad (25–60 %), **3** = casi todo (>60 %), **s** = no se distingue, **q** = salir (se guarda lo hecho). La oscuridad sola no cuenta como oclusión.

## Scripts

| Script | Entregable | Qué produce |
|---|---|---|
| `src/descargar_datos.py` | — | Anotaciones, imágenes CCT-20 y vacías de otras ubicaciones (reanudable) |
| `src/explorar.py` | — | Especies, splits, ubicaciones, tamaño de cajas, hora de captura |
| `src/convertir_yolo.py` | 1 | Dataset YOLO con exclusiones documentadas y fracción de vacías configurable |
| `run_all.py` (pasos 4–7) | 1, 2 | E1/E2/E3, comparación de vacías, mAP por especie cis/trans, falsos disparos en test |
| `src/falsos_disparos_val.py` | 1 | Falsos disparos de un modelo en las vacías de validación |
| `src/predecir.py` | — | Predicciones de todo `cis_test` y `trans_test` en un CSV (las usan los demás) |
| `src/analisis_desbalance.py` | 2 | AP por especie frente al número de cajas de entrenamiento |
| `src/eigencam.py` | 3 | EigenCAM propio sobre el backbone y lift |
| `src/prueba_borrado.py` | 3 | Detección tras tapar al animal |
| `src/verificaciones.py` | 2, 3, 4 | Spearman con p-valor, concentración por cámara, cajas del borde, acuerdo de oclusión |
| `src/etiquetar_oclusion.py` | 4 | Herramienta para etiquetar oclusión |
| `src/tamano_oclusion.py` | 4 | Recall por tamaño y por oclusión, cis frente a trans, y kappa |
| `src/monitoreo.py` | 5 | Reporte por cámara: eventos, especies, individuos, actividad, modelo frente a revisión manual |

## Resultados principales (modelo E3, YOLOv8s, 640 px)

| | mAP@0.5 | mAP@0.5:0.95 | Recall | Falsos disparos en vacías |
|---|---|---|---|---|
| `cis_test` | 0,721 | 0,463 | 0,66 | 3,6 % |
| `trans_test` | 0,459 | 0,286 | 0,42 | 4,3 % |
