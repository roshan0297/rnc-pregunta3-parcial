# Pregunta 3 — Detección de fauna en cámaras trampa (YOLO, CCT-20)

## Entorno

```bash
pip install -r requirements.txt
```

## Pasos

```bash
# 1. Datos: anotaciones + imágenes CCT-20 (~6 GB) + vacías de otras ubicaciones
python src/descargar_datos.py --vacias-extra 3000

# 2. Exploración (tablas y figuras en results/exploracion/)
python src/explorar.py --imgs data/raw/imagenes

# 3. Conversión a formato YOLO (data/yolo/)
python src/convertir_yolo.py --imgs data/raw/imagenes --frac-vacias 0.2
```

## Entrenamiento en Colab / Kaggle

`notebooks/entrenamiento_colab_kaggle.ipynb` hace descarga → exploración → conversión → E1/E2 (efecto de las vacías) → E3 (modelo principal) → evaluación en `cis_test` y `trans_test` → análisis.

## Análisis

| Script | Entregable | Qué produce |
|---|---|---|
| `src/predecir.py` | — | Predicciones de todo `cis_test` y `trans_test` en un CSV (las usan los demás) |
| `src/analisis_desbalance.py` | 2 | AP por especie vs nº de cajas de entrenamiento |
| `src/eigencam.py` | 3 | EigenCAM propio sobre el backbone y medida de concentración en el animal (lift) |
| `src/prueba_borrado.py` | 3 | % de imágenes donde se sigue prediciendo la especie tras tapar al animal |
| `src/etiquetar_oclusion.py` | 4 | Herramienta local para etiquetar oclusión (baja/media/alta) de 300 cajas |
| `src/tamano_oclusion.py` | 4 | Recall por tamaño de caja y por nivel de oclusión, cis vs trans |
| `src/monitoreo.py` | 5 | Reporte por cámara: eventos, especies, individuos, actividad por hora/periodo, modelo vs real |

Pendiente: `run_all.py`.
