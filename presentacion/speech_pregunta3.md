# Speech · Pregunta 3 (≈ 8 minutos)

25 minutos para las tres preguntas → unos **8 minutos para la pregunta 3**. Son 22 diapositivas, así que no todas se explican: algunas se pasan en pocos segundos (marcadas con ⏩). Ritmo de referencia: unas 130 palabras por minuto.

| Bloque | Diapositivas | Tiempo | Acumulado |
|---|---|---|---|
| 1. Problema y datos | 1–5 | 1:30 | 1:30 |
| 2. Preparación y entrenamiento (E1) | 6–8 | 1:15 | 2:45 |
| 3. Resultados y desbalance (E2) | 9–11 | 1:15 | 4:00 |
| 4. ¿Aprendió el fondo? (E3) | 12–14 | 1:30 | 5:30 |
| 5. Robustez (E4) | 15–16 | 0:45 | 6:15 |
| 6. Monitoreo (E5) | 17–19 | 1:00 | 7:15 |
| 7. Ética y cierre (E6) | 20–22 | 0:45 | 8:00 |

---

## Bloque 1 · Problema y datos (1:30)

**[Diap. 1 · Portada]**
La pregunta 3 trata de cámaras trampa: detectar y clasificar fauna con YOLO, y sobre todo medir si el modelo sirve en una cámara que nunca vio.

**[Diap. 2 · Agenda] ⏩**
Las etiquetas E1 a E6 marcan qué entregable responde cada parte.

**[Diap. 3 · El problema]**
Una cámara trampa dispara con movimiento o calor y deja ráfagas de fotos. Muchas son de noche, en infrarrojo, y muchas no tienen ningún animal: las dispara el viento o una sombra. En estas dos fotos el animal ocupa muy poco de la imagen; al gato de la derecha, de hecho, el modelo no lo detectó. La pregunta clave es esta: ¿el detector funciona en una cámara recién instalada, o solo aprendió el fondo de las cámaras con las que se entrenó?

**[Diap. 4 · Datos]**
Usamos CCT-20: casi 58 mil fotos de 20 cámaras y 14 especies. Dos cosas condicionan todo. Primero, el desbalance: la zarigüeya tiene más de 2 500 cajas y el zorro, cinco. Segundo, el split de entrenamiento no trae ninguna foto vacía.

**[Diap. 5 · cis y trans]**
Los splits separan cámaras: *cis* son las mismas diez cámaras del entrenamiento y *trans* son nueve cámaras distintas. Comprobamos que no comparten ninguna ubicación. Eso simula una cámara nueva. Ojo con un detalle: el 74 % de los gatos de entrenamiento viene de una sola cámara.

---

## Bloque 2 · Preparación y entrenamiento — E1 (1:15)

**[Diap. 6 · Preparación]**
Cada exclusión tiene un motivo. Las fotos con especie pero sin caja no sirven ni como positivo ni como fondo: le enseñarían al modelo a ignorar un animal. *Car* no es fauna. Y como train no tiene vacías, agregamos 2 286 fotos vacías de 106 ubicaciones que no son de CCT-20, para no filtrar información del test.

**[Diap. 7 · Entrenamiento] ⏩**
YOLOv8 desde pesos COCO. Cada hiperparámetro responde a los datos: sin volteo vertical porque los animales no aparecen de cabeza, y saturación baja porque más de la mitad de las fotos son infrarrojas.

**[Diap. 8 · Efecto de las vacías]**
E1 y E2 son idénticos salvo por las vacías. Con vacías, los falsos disparos en fotos vacías bajan casi a la mitad: de 6 a 3,4 % en cis y de 8,2 a 4,9 % en trans. Y, lo inesperado, también sube el mAP. Por eso el modelo final, E3, se entrenó con vacías.

---

## Bloque 3 · Resultados y desbalance — E2 (1:15)

**[Diap. 9 · cis frente a trans]**
Este es el resultado central. En las cámaras conocidas el mAP a 0,5 es 0,72; en cámaras nuevas baja a 0,46: 26 puntos menos. Lo importante es de dónde viene la caída: el recall baja de 0,66 a 0,42, pero los falsos disparos casi no cambian. En una cámara nueva, el modelo deja de ver animales mucho más de lo que inventa.

**[Diap. 10 · Por especie]**
La caída no es pareja. Zarigüeya y zorrillo casi se mantienen; gato, ave, roedor y ardilla se desploman. El gato pasa de 0,93 a 0,29.

**[Diap. 11 · Desbalance]**
¿Es por el desbalance? Solo en parte. En cis, más cajas significa más AP: Spearman de 0,68, significativo. En trans baja a 0,49 y deja de serlo. El gato tiene más de mil cajas y aun así se desploma. En cámaras nuevas importa más la variedad de escenas que la cantidad.

---

## Bloque 4 · ¿Aprendió el fondo? — E3 (1:30)

**[Diap. 12 · Concentración por cámara]**
Las tres especies que más caen, roedor, gato y ave, son las que en entrenamiento aparecieron en solo dos cámaras. Pero con las once especies la correlación es débil. No basta para afirmar que aprendió el fondo, así que lo probamos directamente.

**[Diap. 13 · Prueba de borrado]**
Si el modelo usara el fondo como atajo, al tapar al animal seguiría prediciendo la especie. Tapamos la caja en 600 imágenes y volvimos a detectar. En ninguna apareció la especie en la zona tapada. Solo en tres apareció en otra parte de la escena. El modelo no inventa la especie a partir del escenario.

**[Diap. 14 · EigenCAM]**
Para la interpretabilidad usamos EigenCAM, la variante de Grad-CAM sin gradientes, más fácil de aplicar a un detector. El mapa se concentra en la escena, no en el animal, incluso en los aciertos. Parece contradecir el borrado, pero responden preguntas distintas: EigenCAM toma la dirección de mayor variación, y en una cámara fija eso es el fondo. El backbone representa sobre todo la escena, pero la decisión no depende de ella. Conclusión: la caída en trans no es un atajo, es que el modelo no reconoce al animal cuando cambian el fondo y la luz.

---

## Bloque 5 · Robustez — E4 (0:45)

**[Diap. 15 · Tamaño]**
Los animales pequeños fallan en ambos tests, como se esperaba. Lo inesperado son los grandes en cámaras nuevas: el modelo encuentra el 64 % pero acierta la especie solo en el 46 %. Revisamos esos errores: el 80 % toca el borde de la imagen. Son animales tan cerca que salen cortados, y sin la silueta completa el modelo confunde mapache con coyote.

**[Diap. 16 · Oclusión]**
CCT-20 no anota la oclusión, así que armamos una muestra de 300 cajas y le pusimos a cada una un nivel según qué parte del animal no se ve. El patrón es claro: con oclusión baja el recall es 0,91 en cis y 0,82 en trans; con oclusión media baja a 0,74 y 0,52. Ver solo una parte del animal ya es difícil, y en una cámara nueva lo es mucho más.

*Si preguntan cómo se etiquetó:* "Por tiempo, el etiquetado inicial lo hizo un modelo de visión siguiendo el criterio, y una persona re-etiqueta 50 cajas a ciegas para medir el acuerdo con kappa de Cohen". Dilo así: es lo que figura en el informe.

---

## Bloque 6 · Monitoreo — E5 (1:00)

**[Diap. 17 · Reglas] ⏩**
Un ecólogo no necesita cajas por foto, sino eventos, especies e individuos. Agrupamos fotos en eventos de 60 segundos y contamos individuos como el máximo por foto, no la suma, porque el mismo animal sale en varias fotos de la ráfaga.

**[Diap. 18 · Cámara 46]**
Lo probamos en la cámara 46, que el modelo nunca vio. Acierta la especie en tres de cada cuatro eventos y, lo más útil, reproduce los patrones de actividad: la proporción nocturna de zarigüeya, mapache, zorrillo y lince difiere como mucho en cinco centésimas.

**[Diap. 19 · Dónde falla]**
Pero no reemplaza a una persona. Pierde el 19 % de los eventos con animal, casi no registra aves ni roedores, y aparecen 55 gatos donde había 5: en realidad eran linces, coyotes y mapaches.

---

## Bloque 7 · Ética y cierre — E6 (0:45)

**[Diap. 20 · Ética]**
El error no se reparte por igual: las especies raras, que son las que más importan para la conservación, son las peor detectadas. Un reporte automático podría dar por desaparecida a una especie que solo es invisible para el modelo. Por eso proponemos un humano en el circuito, además de proteger la ubicación de especies sensibles.

**[Diap. 21 · Conclusiones]**
Cuatro ideas: las vacías como fondo valen la pena; en cámaras nuevas se pierde recall, no precisión; no es un atajo de fondo, es falta de variedad de escenas; y como herramienta sirve para patrones de actividad, no para inventarios completos.

**[Diap. 22 · Reproducibilidad] ⏩**
Todo se reproduce con un solo comando, `python run_all.py`. Gracias.

---

## Si preguntan

- **¿Qué es mAP?** El promedio, entre las 14 especies, del área bajo la curva precisión–recall. A 0,5 significa que la caja predicha debe solaparse al menos a la mitad con la real.
- **¿Por qué no Grad-CAM?** Grad-CAM necesita un único puntaje del que derivar gradientes; un detector produce cientos de cajas. EigenCAM es la variante sin gradientes, aplicable al backbone.
- **¿Por qué umbral 0,5?** Se eligió en validación: el F1 máximo está en 0,45 y la curva es plana entre 0,4 y 0,6. Nunca se ajustó en test.
- **¿Por qué no MegaDetector?** No da la especie, y entre sus datos de entrenamiento está Caltech Camera Traps: contaminaría la evaluación en trans.
- **¿Las vacías no filtran información?** No: vienen de 106 ubicaciones que no están en CCT-20.
- **El pico de mediodía en zarigüeya y mapache (diap. 18):** también aparece en las etiquetas reales, así que no es error del modelo. Lo más probable es un problema del reloj de la cámara (medianoche registrada como mediodía).
