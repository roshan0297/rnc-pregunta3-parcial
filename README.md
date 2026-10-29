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

`notebooks/entrenamiento_colab_kaggle.ipynb` hace descarga → exploración → conversión → E1/E2 (efecto de las vacías) → E3 (modelo principal) → evaluación en `cis_test` y `trans_test`.

Pendiente: evaluación cis/trans, oclusión/tamaño, módulo de monitoreo y `run_all.py`.
