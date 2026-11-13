# Predicción horaria del uso de biciMAD

Entrega 2 de Aprendizaje Automático · 13 de noviembre de 2026  
Sofía Muñiz Garzón · Alberto Caballero Lorente  
Grado en Ingeniería Matemática · Universidad Francisco de Vitoria

El proyecto compara dos referencias históricas, una regresión Ridge y un MLP para predecir cuántos viajes registrados se iniciarán en toda la red durante la siguiente hora. Se utiliza 2022 y se mantiene la separación temporal planteada en la primera entrega. Los resultados de esta versión son preliminares y proceden de una ejecución real de los archivos oficiales.

## Qué contiene esta entrega

El preprocesado de la versión con calendario e historial está completo: lectura de los doce meses, limpieza, selección del ámbito, control de duplicados, tabla horaria, calendario, variables históricas y transformación ajustada solo con entrenamiento. Los primeros modelos se han ejecutado. Se incluye una auditoría de meteorología, que no forma parte de las entradas de esta versión. Quedan pendientes el ajuste de hiperparámetros, el análisis de sensibilidad y la evaluación final.

Noviembre y diciembre están preparados, pero **no se han evaluado**. El programa de esta entrega no contiene una opción para calcular sus métricas por accidente.

## Inicio rápido: repetir los modelos

Se ha comprobado con Python 3.12. Los CSV agregados incluidos permiten repetir los modelos sin descargar los viajes originales. Desde esta carpeta:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m bicimad train
python -m pytest -q
```

En Windows, la activación es `.venv\Scripts\activate`. Si el ejecutable se llama `python` en lugar de `python3`, se utiliza ese nombre en la primera línea. Los cálculos se ejecutan en CPU; no hace falta GPU. La primera importación de las bibliotecas y la creación de la caché de fuentes pueden tardar más que el entrenamiento.

`requirements.txt` fija las dependencias directas y `requirements-lock.txt` fija también las dependencias transitivas del entorno comprobado. El registro `results/run.json` recoge las versiones, variables, configuración, cortes y huellas utilizados. Diferencias de plataforma o bibliotecas numéricas pueden producir pequeñas variaciones en los últimos decimales.

## Reproducción completa desde las fuentes

```bash
python -m bicimad all
```

Este comando descarga o verifica las fuentes, prepara los datos y repite EDA y modelos. El ZIP anual de EMT ocupa unos 301 MiB; hay otros archivos pequeños de calendario y meteorología. Se necesita conexión para la primera descarga y conviene disponer de varios GB de memoria para procesar los identificadores y cada mes. Los ZIP se leen en memoria de uno en uno, sin extraer rutas arbitrarias ni cargar todos los viajes en una única tabla.

También se pueden separar las etapas:

```bash
python -m bicimad download
python -m bicimad prepare
python -m bicimad train
```

`download` verifica cada SHA-256 contra `config/sources.json`. Si una fuente cambia, se detiene: no se debe sustituir la huella sin revisar la nueva versión. `prepare` trabaja sobre los originales ya descargados; para asegurar su procedencia, ejecutar primero `download`, o utilizar directamente `all`. No se necesitan claves ni cuentas privadas.

## Organización

| Ruta | Contenido |
|---|---|
| `bicimad/data.py` | Descarga, control de fuentes, limpieza mensual, calendario y auditoría meteorológica. |
| `bicimad/features.py` | Rejilla temporal, retardos, media móvil y separación por fechas. |
| `bicimad/models.py` | Referencias, imputación, normalización, Ridge, MLP y evaluación de validación. |
| `bicimad/eda.py` | Descriptivos de entrenamiento y gráficos. |
| `config/experiment.json` | Arquitectura, hiperparámetros y semillas del experimento. |
| `config/sources.json` | URLs, licencias, tamaños y SHA-256 de las fuentes consultadas. |
| `data/processed/` | Serie horaria, calendario y tabla de variables, sin identificadores individuales. |
| `results/` | Auditorías, métricas, predicciones, pérdida del MLP y registro de ejecución. |
| `tests/` | Diez pruebas sobre las decisiones que podrían introducir errores o fuga de información. |

Los originales de `data/raw/` y los entornos virtuales se excluyen de Git. Las figuras incluyen el EDA de entrenamiento y un ejemplo de la primera semana completa de septiembre. Ese ejemplo utiliza la semilla 42; los resultados numéricos comparan siempre todo el bloque y las tres semillas.

## Decisiones de preparación

1. Los CSV se leen con separador `;`. Se eliminan las filas completamente vacías y se comprueba que estén las columnas necesarias. El identificador `idTrip` falta en febrero y agosto–diciembre.
2. Se detectan duplicados por `idTrip` cuando existe y por una clave común formada por bicicleta, fecha de salida, fecha de llegada y estaciones. Se conserva la primera aparición, siguiendo el orden de meses y filas. Las claves se almacenan como hashes en memoria y no se publican. La comparación de claves cubre también las fronteras entre meses; la probabilidad de colisión del hash de 64 bits es pequeña, aunque no es una identidad matemática exacta.
3. El ámbito operativo se fija en `fleet=1`, `unlocktype=STATION` y estación de origen positiva y entera. En los archivos inspeccionados la flota 1 solo presenta salidas de estación y la flota 2 mezcla tipos `FREE` y `STATION`. Se utiliza el código observado para delimitar la población, sin atribuirle categorías de usuarios no documentadas.
4. La duración, la estación de destino y la fecha de llegada se revisan para calidad/identidad del registro, pero no entran en las variables predictoras. No se eliminan salidas válidas por una duración atípica: el objetivo es contar salidas registradas.
5. Las fechas sin zona se interpretan como hora local de Madrid. La ausencia de la hora inexistente de marzo es coherente con esta hipótesis. En octubre se excluyen las salidas de la hora repetida y se dejan desconocidas sus dos horas UTC, ya que no hay desplazamiento que permita distinguirlas. El índice final es regular, único y ordenado en UTC; el calendario se calcula en hora local.
6. Las horas sin salidas registradas se dejan como desconocidas, sin imputar el objetivo. Esto es conservador: no se puede distinguir una ausencia real de viajes de una pérdida de información. No garantiza detectar pérdidas parciales. Se conservan los recuentos positivos, también los bajos del episodio de agosto.
7. Para predecir `y[t]`, que cuenta las salidas de `[t,t+1)`, se utilizan `y[t-1]`, `y[t-24]`, `y[t-168]` y la media de las 24 horas terminadas antes de `t`. Esta media se calcula después de desplazar la serie una hora. Los retardos miden horas transcurridas; alrededor del cambio de horario pueden diferir de la misma hora local del día o semana anterior.
8. Hora, día de semana y mes se codifican con seno y coseno. Se añaden festivo y fin de semana. Son doce variables antes de los indicadores de ausencia. Las primeras 168 horas se excluyen. Los huecos de variables se imputan mediante la mediana de entrenamiento; los objetivos desconocidos se excluyen de ajuste y evaluación.

`quality_monthly.csv` contiene el balance mensual. Las exclusiones son sucesivas, por lo que se pueden sumar sin contar dos veces la misma fila. Las columnas de duración son diagnósticos sobre los registros retenidos, no exclusiones. `unknown_hours.csv` enumera los cinco objetivos desconocidos: tres horas de agosto y las dos horas repetidas de octubre.

## Protocolo experimental

| Bloque | Periodo local de 2022 | Horas útiles |
|---|---|---:|
| Entrenamiento | Enero–agosto | 5.660 |
| Validación | Septiembre–octubre | 1.463 |
| Prueba, sin evaluar | Noviembre–diciembre | 1.464 |

Las referencias usan la hora anterior y 168 horas antes; si falta esa observación, recurren a la mediana de entrenamiento por día de semana y hora, y después a la global. Los modelos reciben medianas e indicadores de ausencia mediante `SimpleImputer`. El escalador se ajusta exclusivamente con entrenamiento. El MLP normaliza también el objetivo con `TransformedTargetRegressor`, que devuelve las predicciones a la escala de viajes.

La evaluación simula predicciones sucesivas a una hora, con acceso a recuentos anteriores observados, sin reentrenamiento durante validación. No equivale a pronosticar dos meses de una sola vez. Las predicciones negativas se recortan a cero y no se redondean. Los seis experimentos se evalúan sobre las mismas horas válidas.

El MLP tiene una única capa de 32 neuronas, ReLU, Adam, lotes de 64, tasa de aprendizaje 0,001 y regularización 0,0001. Se usa `shuffle=False`, `early_stopping=False`, `max_iter=400`, `tol=0.0001` y `n_iter_no_change=20`. La parada depende de la pérdida de entrenamiento, no de una partición aleatoria ni de escoger la mejor época de validación. Se repite la misma configuración con las semillas 7, 21 y 42, sin seleccionar una como resultado único.

## Resultados incluidos

| Modelo | MAE de validación | RMSE de validación |
|---|---:|---:|
| Hora anterior | 168,51 | 242,58 |
| 168 horas antes | 117,38 | 204,37 |
| Ridge, alpha 1 | **105,77** | 156,05 |
| MLP, semilla 7 | 109,72 | **148,71** |
| MLP, semilla 21 | 130,13 | 174,24 |
| MLP, semilla 42 | 123,46 | 167,91 |

El MAE medio del MLP es 121,10, con desviación muestral entre semillas de 10,41. Estas tres repeticiones no proporcionan un intervalo de confianza. Ridge obtiene el menor MAE, de modo que el avance no confirma la hipótesis de mejora del 10 %. Los archivos CSV guardan las métricas con su precisión completa y las predicciones de cada hora.

## Meteorología y tareas siguientes

`weather_coverage.csv` audita los doce archivos oficiales. Solo considera válidas las parejas `Hxx/Vxx` con marca `V` y valor numérico. Su denominador es estaciones presentes en el mes × días del mes × 24; incluye la falta de días completos, pero no estaciones que desaparezcan del mes entero. Por eso deben leerse conjuntamente cobertura y número de estaciones. Los códigos relevantes son 81 (viento, m/s), 83 (temperatura, °C) y 89 (precipitación, l/m²), según el intérprete municipal.

La meteorología queda fuera del modelo de esta entrega. Para añadirla se necesitaría fijar estaciones comparables y resolver la alineación de H01–H24 y su zona horaria antes de construir observaciones anteriores a la predicción. La versión actual con calendario e historial no depende de esas decisiones.

Entre el 14 y el 20 de noviembre se prevé ajustar pocas configuraciones, comparar la estabilidad del MLP y estudiar la sensibilidad al episodio de actividad muy baja de agosto. Una vez cerrada la selección, se reajustará con enero–octubre y se evaluará noviembre–diciembre. El informe detalla prioridades, responsables y fechas. No se ha implementado todavía la selección final ni se han calculado resultados de prueba.

## Comprobaciones

`python -m pytest -q` comprueba causalidad de variables, rejilla y cambios de hora, duplicados entre archivos, ausencia de fechas en calendario, cortes temporales, valores de respaldo, ajuste de transformaciones y exclusión de prueba en las predicciones guardadas. Las pequeñas entradas sintéticas de las pruebas son solo casos de comprobación; no se emplean para entrenar ni para calcular los resultados del informe.

## Fuentes y condiciones de reutilización

- [EMT Madrid: históricos de biciMAD 2017–2023](https://datos.emtmadrid.es/dataset/historicos-de-bicimad-2017_2023). Sus metadatos indican Open Data Commons Attribution License. La tabla de salidas agregadas es una transformación de estos datos y no es una publicación oficial de EMT.
- [Ayuntamiento de Madrid: calendario laboral](https://datos.madrid.es/dataset/300082-0-calendario_laboral). CC BY 4.0. Se seleccionan las fechas de 2022 y se deriva el indicador de festivo.
- [Ayuntamiento de Madrid: meteorología horaria](https://datos.madrid.es/dataset/300352-0-meteorologicos-horarios). CC BY 4.0. La auditoría de cobertura es una elaboración propia.
- [Intérprete municipal de los datos meteorológicos](https://datos.madrid.es/dataset/300352-0-meteorologicos-horarios/resource/300352-122-meteorologicos-horarios/download/300352-122-meteorologicos-horarios.pdf).
- Documentación oficial de [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) y [MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html).

Consulta y descarga: 13 de noviembre de 2026. Los originales se descargan con el programa y no están incluidos en el repositorio. No se asigna una nueva licencia a las fuentes ni se ha decidido una licencia pública para el código; esa decisión corresponde a sus autores.

## Versión de la entrega

[Repositorio del proyecto](https://github.com/sofiamgarz/Proyecto-Final-BiciMAD). Esta primera versión reúne el código, las tablas agregadas, los resultados y la documentación de la entrega 2. El informe se entrega por separado. Los archivos originales y los entornos virtuales quedan excluidos mediante `.gitignore`.

## Uso de herramientas de apoyo

Se utilizó Codex de OpenAI, basado en GPT-6, como apoyo a la implementación, ejecución, documentación y preparación del informe. Los resultados proceden de los datos oficiales y de las ejecuciones guardadas; la revisión de los supuestos y la comprensión del código siguen siendo responsabilidad de los autores. Se incluye esta declaración para mantener la coherencia con la guía de la asignatura.
