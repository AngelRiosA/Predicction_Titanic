# Predicción de Supervivencia en el Titanic
Este proyecto utiliza modelos de aprendizaje automático para predecir si un pasajero del Titanic sobrevivió o no, basándose en características como su edad, sexo, tipo de billete y otros factores.

## Descripción General
El naufragio del Titanic es un evento histórico trágico que cobró la vida de miles de personas. En este proyecto, buscamos aplicar técnicas de ciencia de datos para analizar el conjunto de datos de pasajeros del Titanic y construir modelos que puedan predecir la probabilidad de supervivencia de un pasajero en función de sus características.

## Conjunto de Datos
El conjunto de datos utilizado en este proyecto proviene de Kaggle y contiene información sobre los pasajeros del Titanic, incluyendo:

*   **Survived**: Variable objetivo que indica si el pasajero sobrevivió (1) o no (0).
*   **Pclass**: Clase del billete (1 = primera clase, 2 = segunda clase, 3 = tercera clase).
*   **Sex**: Sexo del pasajero (masculino o femenino).
*   **Age**: Edad del pasajero.
  
## Metodología
1.  **Análisis Exploratorio de Datos (EDA)**: Se realizó un análisis exhaustivo del conjunto de datos para comprender las relaciones entre las variables y la supervivencia. Se utilizaron técnicas de visualización y estadística descriptiva.
2.  **Ingeniería de Características**: Se crearon nuevas características a partir de las existentes para mejorar el rendimiento de los modelos.
3.  **Selección de Modelos**: Se probaron varios modelos de clasificación, incluyendo regresión logística, árboles de decisión y bosques aleatorios.
4.  **Evaluación de Modelos**: Se evaluó el rendimiento de los modelos utilizando métricas como precisión, exactitud, F1-score y AUC-ROC.
5.  **Ajuste de Hiperparámetros**: Se optimizaron los hiperparámetros de los modelos para mejorar su rendimiento.

## Resultados
Los modelos de aprendizaje automático lograron una precisión del X% en la predicción de la supervivencia de los pasajeros del Titanic. Los resultados muestran que características como la clase del billete, el sexo y la edad influyen significativamente en la probabilidad de supervivencia.

## Despliegue
Este proyecto ha sido desplegado y está disponible para su uso en [enlace a la aplicación desplegada]. Puedes acceder a la aplicación y probarla ingresando la información de un pasajero para obtener una predicción de su probabilidad de supervivencia.
