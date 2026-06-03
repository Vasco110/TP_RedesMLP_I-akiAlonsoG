
# Preguntas sobre el ejemplo de clasificación de imágenes con PyTorch y MLP

## 1. Dataset y Preprocesamiento
- ¿Por qué es necesario redimensionar las imágenes a un tamaño fijo para una MLP?
   Respuesta: Es necesario estandarizar el tamaño porque la primera etapa de la red tiene una cantidad fija de entradas 
- ¿Qué ventajas ofrece Albumentations frente a otras librerías de transformación como `torchvision.transforms`?
   Respuesta: Albumentations es una libreria más rápida, que ofrece una programación más directa sin tener que modificar por separado las máscaras y las imágenes en sí. Además Albumentations tiene transformaciones mejores en calidad y complejidad, 'torchvision.transforms' utiliza únicamente transformaciones de algebra lineal.
- ¿Qué hace `A.Normalize()`? ¿Por qué es importante antes de entrenar una red?
   Respuesta: Lo que hace es permite usar la distubión normal estándar modificando el valor original restándole la media y dividiendo por el desvío de la muestra. Es necesario porque la diferencia hace que los valores más bajos tengan velocidades más bajas de entrenamiento, demorando entonces el entrenamiento de la red total.  
- ¿Por qué convertimos las imágenes a `ToTensorV2()` al final de la pipeline?
   Respuesta: Es importante convertirlo a torch.Tensor porque son los tipos de datos que reciben las redes que se entrenaran a continuación. Se implementa luego de las tranformaciones y la normalización porque las funciones utilizadas trabajan sobre la imágen y la máscara, y no pueden ser aplicadas a tensores. Este tipo de dato permite trabajar con la GPU para optimizar el procesamiento de la red.
## 2. Arquitectura del Modelo
- ¿Por qué usamos una red MLP en lugar de una CNN aquí? ¿Qué limitaciones tiene?
   Respuesta: Utilizamos MLP porque es rápido de entrenar y requiere menos computo y memoria (pudiendo correr el modelo incluso en laptops) y funciona muy bien para problemas de regresión y clasificación si no se trata de un patrón demasiado complejo. Es útil este modelo para aprender el funcionamiento de las redes neuronales.
   La complejidad de las redes MLP crece rápidamente cuando se aumenta el tamaño de la muestra. Además pierde la referencia espacial de las imágenes al tener que procesar los datos en una matriz 1D. 
- ¿Qué hace la capa `Flatten()` al principio de la red?
   Respuesta: Pasa los datos de la imágen a un tensor 1D para poder ser procesado en la red en uso. Si usasemos una red CNN podríamos usar la referencia espacial de las imágenes trabajando en matrices 2D o 3D.
- ¿Qué función de activación se usó? ¿Por qué no usamos `Sigmoid` o `Tanh`?
   Respuesta: Se utilizó una función ReLU (Unidad Lineal Rectificada). La prpincipal razón es porque genera 0 para valores negativos, haciendo que el entrenamiento de la red sea más eficiente y rápido. Evita el desvanecimiento de gradiente, porque durante el backpropagation los resultados de los gradientes de las primeras capas no se multiplican por valores menores a 1.
- ¿Qué parámetro del modelo deberíamos cambiar si aumentamos el tamaño de entrada de la imagen?
   Respuesta: input_size que deberá ser el alto por el ancho de la imágen por 3 debido a los colores (r,g,b).

## 3. Entrenamiento y Optimización
- ¿Qué hace `optimizer.zero_grad()`?
   Respuesta: Resetea los valores de gradientes guardados entre iteraciones de entrenamiento.
- ¿Por qué usamos `CrossEntropyLoss()` en este caso?
   Respuesta: Porque es una función de loss en la que el gradiente es muy alto cuando la red es muy mala, entonces acelera el entrenamiento. Sin embargo el gradiente se vuelve mas bajo al acercarse al minimo absoluto.
- ¿Cómo afecta la elección del tamaño de batch (`batch_size`) al entrenamiento?
   Respuesta: Un batch_size grande hace que el entrenamiento de la red sea más rápido con gradientes más estables y precisos pero consume más cantidad de memorias. Por otro lado batch_size chicos ayudan a evitar el overfitting, ya que es más díficil que se sobre-ajuste porque evita que el modelo converja a minímos locales.
- ¿Qué pasaría si no usamos `model.eval()` durante la validación?
   Respuesta: Si no lo usasemos seguiría entrenando y actualizando los gradientes con datos de validación, por lo que no serían válidos a largo plazo los resultados de las validaciones, es lo mismo que no haber separado el dataset, ya que al validar seguiria entrenando.

## 4. Validación y Evaluación
- ¿Qué significa una accuracy del 70% en validación pero 90% en entrenamiento?
  Respuesta: Probablemente se deba a un caso de Overfitting, donde la red se ajusta excesivamente al dataset de train pero no es bueno para generalizar a otros datos dentro del mismo dominio.
- ¿Qué otras métricas podrían ser más relevantes que accuracy en un problema real?
  Respuesta: En algunos problemas reales, como deteccion de enfermedades podria ser muy interesante analizar falsos positivos, error tipo 1, y positivos no detectados, error tipo 2,  por separado, ya que en casos de enfermedades por ejemplo seria mucho mas grave no detectar que alguien esta enfermo, a sugerir estudios mas precisos a un falso positivo. Esto se hace con una matriz de confusion.
- ¿Qué información útil nos da una matriz de confusión que no nos da la accuracy?
  Respuesta: Justamente como se menciono anteriormente, otorga informacion importante sobre ambos tipos de errores, ademas de cuantos casos positivos o negativos se detectaron correctamente.
- En el reporte de clasificación, ¿qué representan `precision`, `recall` y `f1-score`?
  Respuesta: Precision es la cantidad de casos que la red designo como positivos, que porcentaje realmente lo eran. Recall refiere a cuantos de los verdaderos positivos, la red es capaz de detectar. Por ultimo el F1-score es una forma de unificar ambas metricas en una, utilizando una media armonica, el doble de la multiplicacion sobre la suma.

## 5. TensorBoard y Logging
- ¿Qué ventajas tiene usar TensorBoard durante el entrenamiento?
  Respuesta: Tensorboard permite visualizar el entrenamiento de la red, ofreciendo monitoreo en tiempo real de las metricas, aun durante la duracion del script de entrenamiento, esto puede ayudar a detectar fallas antes de que el programa termine para no perder tiempo en entrenar una red mala, caso de overfitting o vanishing gradient. 
- ¿Qué diferencias hay entre loguear `add_scalar`, `add_image` y `add_text`?
  Respuesta: La diferencia radica en que se busca graficar, como sus nombres indican, no es lo mismo graficar un escalar, que se puede graficar su evolucion en el tiempo en una recta, que graficar imagenes, la cual es una para cada tiempo y se requeriria graficar mas de una o un slider para poder observar la evolucion o guardar un texto, que muestre algun dato de forma ordenada, pero no graficado. 
- ¿Por qué es útil guardar visualmente las imágenes de validación en TensorBoard?
  Respuesta: Ademas de que permite verificar la evolucion de la red de forma visual, aporta mayor informacion que un simple porcentaje de resultado, ya que se podria llegar a detectar algun patron en los fallos, o visualizar directamente casos de overfitting.
- ¿Cómo se puede comparar el desempeño de distintos experimentos en TensorBoard?
  Respuesta: Tensorboard permite guardar carpetas separadas dentro del mismo proyecto o directorio, y despues superponer los graficos de los mismos para compararlos visualmente, con todas las ventajas mencionadas anteriormente.

## 6. Generalización y Transferencia
- ¿Qué cambios habría que hacer si quisiéramos aplicar este mismo modelo a un dataset con 100 clases?
  Respuesta: En un principio se podria aumentar la cantidad de salidas a 100, siendo cada una la probabilidad de cada clase, sin embargo, es probable que esto resulte en valores muy bajos, se podria optimizar utilizando distintas etapas por ejemplo, de forma de separar primero entre grupos de clases con ciertas similitudes, y despues clasificar dentro de estos subgrupos.
- ¿Por qué una CNN suele ser más adecuada que una MLP para clasificación de imágenes?
  Respuesta: Principalmente las MLP al realizar el `Flatten()` pierde relaciones de dimensionalidad. Analiza puramente de forma matematica, mientras que las redes convolucionales podrian detectar patrones similares en distintos puntos de una imagen, Por ejemplo si clasificamos perros y gatos, una CNN puede detectar un mismo gato este en una esquina o en el centro de la imagen, y a las MLP se les complica mas. Ademas las imagenes de alta calidad terminan teniendo una gran cantidad de entradas y por lo tanto de neuronas en MLP mientras que en CNN, utilizando entonces menor cantidad de parametros a entrenar.
- ¿Qué problema podríamos tener si entrenamos este modelo con muy pocas imágenes por clase? 
  Respuesta: el problema mas comun al tener un dataset chico es el overfitting, ya que si es un modelo con muchos parametros y un dataset chico la red tiende a utilizar los parametros extra para generar el overfitting, en lugar de generalizar de la mejor manera posible, estancandose en minimos locales.
- ¿Cómo podríamos adaptar este pipeline para imágenes en escala de grises?
  Respuesta: Habria que cambiar la cantidad de entradas, ya que antes por cada pixel teniamos 3 entradas, rojo azul y verde y ahora solo el brillo de la escala de grises

## 7. Regularización

### Preguntas teóricas:
- ¿Qué es la regularización en el contexto del entrenamiento de redes neuronales?
  Respuesta: Son metodos para mejorar los resultados de las redes, reduciendo o evitando el overfitting para mejorar la generalizacion de una red entrenada.
- ¿Cuál es la diferencia entre `Dropout` y regularización `L2` (weight decay)?
  Respuesta: Son diferentes metodos, L2 penaliza los pesos altos en el gradiente, mientras que dropout consiste en realizar cada prediccion de tren con un subset de neuronas, para evitar la dependencia de neuronas predominantes.
- ¿Qué es `BatchNorm` y cómo ayuda a estabilizar el entrenamiento?
  Respuesta: Batch Normalization consiste en normalizar las salidas tras cada batch, para deshacer el Internal Covariate Shift que termina limitando el LR que se puede utilizar. Es una tecnica para acelerar el entrenamiento, que termina teniendo algunos efectos secundarios de regularizacion.
- ¿Cómo se relaciona `BatchNorm` con la velocidad de convergencia?
  Respuesta: Se utiliza para acelerar el entrenamiento en redes multietapa de la misma forma que la normalizacion funcionaba en redes de una unica capa. Ayuda a evitar el vanishing gradient
- ¿Puede `BatchNorm` actuar como regularizador? ¿Por qué?
  Respuesta: Tiene un efecto de regularizacion ya que la normalizacion genera un leve ruido en los datos, generando un efecto como si hubiera mas observaciones, cercanas, evitando umbrales de decision de alta precision.
- ¿Qué efectos visuales podrías observar en TensorBoard si hay overfitting?
  Respuesta: umbrales de decision de alta precision y zonas muy finas de indecision. Ademas deberiamos ver discrepancias altas entre train y validacion, con una evolucion en el tiempo donde validacion termina creciendo.
- ¿Cómo ayuda la regularización a mejorar la generalización del modelo?
  Respuesta: Basicamente la regularizacion fuerza a empeorar levemente los resultados de train de forma de evitar el sobreajuste, por lo tanto el entrenamiento termina forzado a buscar buenas generalizaciones.

### Actividades de modificación:
1. Agregar Dropout en la arquitectura MLP:
   - Insertar capas `nn.Dropout(p=0.5)` entre las capas lineales y activaciones.
   - Comparar los resultados con y sin `Dropout`.

2. Agregar Batch Normalization:
   - Insertar `nn.BatchNorm1d(...)` después de cada capa `Linear` y antes de la activación:
     ```python
     self.net = nn.Sequential(
         nn.Flatten(),
         nn.Linear(in_features, 512),
         nn.BatchNorm1d(512),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(512, 256),
         nn.BatchNorm1d(256),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(256, num_classes)
     )
     ```

3. Aplicar Weight Decay (L2):
   - Modificar el optimizador:
     ```python
     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
     ```

4. Reducir overfitting con data augmentation:
   - Agregar transformaciones en Albumentations como `HorizontalFlip`, `BrightnessContrast`, `ShiftScaleRotate`.

5. Early Stopping (opcional):
   - Implementar un criterio para detener el entrenamiento si la validación no mejora después de N épocas.

### Preguntas prácticas:
Analisis Dropout: Por alguna razon empeoro gravemente la red al incluir el dropout con pocos epochs, esto se debe a que se vuelven muy pocas neuronas las de entrenamiento al apagar el 50% en esta red, ralentizando fuertemente el entrenamiento.
- ¿Qué efecto tuvo `BatchNorm` en la estabilidad y velocidad del entrenamiento?
  Respuesta: Nos permite utilizar un learning rate mas alto, ademas mejoro los resultados de valoracion gracias a mantener estabilidad de parametros entre las capas
- ¿Cambió la performance de validación al combinar `BatchNorm` con `Dropout`?
  Respuesta: Si, no solo empeoro con respecto a utilizar BatchNorm en solitario, aunque mejoro con respecto a solo Dropout, sino que ralentizo mucho la convergencia. 
- ¿Qué combinación de regularizadores dio mejores resultados en tus pruebas?
  Respuesta: BatchNorm con early stop y data augmentation.
- ¿Notaste cambios en la loss de entrenamiento al usar `BatchNorm`?
  Respuesta: Mejoro notablemente de 1.5 a 1, utilizando early stop para ambos y 100 epochs, no completados en ninguno de los 2 casos.

## 8. Inicialización de Parámetros

### Preguntas teóricas:
- ¿Por qué es importante la inicialización de los pesos en una red neuronal?
  Respuesta: Ayuda a mantener aproximadamente la media y desvio estandar en cada capa al inicio, para poder mantener el LR en toda la red. Para ello se busca que la media de los pesos tambien sea 0 asi la media de la salida tambien es 0. La funcion de activacion tambien debe ser considerada. Sirve para acelerar el inicio del entrenamiento, evitando tener vanishing o exploding gradient al inicio del entrenamiento, Al mantener la covarianza entre la salida y la entrada. Mejora la eficiencia de uso del LR.
- ¿Qué podría ocurrir si todos los pesos se inicializan con el mismo valor?
  Respuesta: Se busca romper la simetria entre las unidades ocultas por que sino las salidas son iguales, con el mismo gradiente y terminas teniendo neuronas duplicadas, Y es una simetria que no se va a romper nunca. 
- ¿Cuál es la diferencia entre las inicializaciones de Xavier (Glorot) y He?
  Respuesta: Xavier-Busca igualar varianza de entrada y de salida, funciona con activaciones sigmoidea y tangente hiperbolica.
            He-Optimizada para ReLU, busca compensar el efecto de la ReLU de perdida de la mitad de la varianza generada por la activacion que apaga la mitad de las neuronas con salidas menores a 0, compensa este calculo multiplicando por 2 la varianza, generando pesos ligeramente mas grandes para compensar esta perdida.
- ¿Por qué en una red con ReLU suele usarse la inicialización de He?
  Respuesta: Justamente por lo que se menciono, que compensa el efecto sobre la varianza de la funcion de activacion
- ¿Qué capas de una red requieren inicialización explícita y cuáles no?
  Respuesta: Es necesario inicializar todas las capas que contengan parametros entrenables mediante el backpropagation, que necesitan ser inicializadas para evitar vanishing o expoding gradient

### Actividades de modificación:
1. Agregar inicialización manual en el modelo:
   - En la clase `MLP`, agregar un método `init_weights` que inicialice cada capa:
     ```python
     def init_weights(self):
         for m in self.modules():
             if isinstance(m, nn.Linear):
                 nn.init.kaiming_normal_(m.weight)
                 nn.init.zeros_(m.bias)
     ```

2. Probar distintas estrategias de inicialización:
   - Xavier (`nn.init.xavier_uniform_`)
   - He (`nn.init.kaiming_normal_`)
   - Aleatoria uniforme (`nn.init.uniform_`)
   - Comparar la estabilidad y velocidad del entrenamiento.

3. Visualizar pesos en TensorBoard:
   - Agregar esta línea en la primera época para observar los histogramas:
     ```python
     for name, param in model.named_parameters():
         writer.add_histogram(name, param, epoch)
     ```

### Preguntas prácticas:
- ¿Qué diferencias notaste en la convergencia del modelo según la inicialización?
  Respuesta: La inicializacion por el metodo de he converge de forma mas rapida al metodo de xavier
- ¿Alguna inicialización provocó inestabilidad (pérdida muy alta o NaNs)?
  Respuesta: Algunos casos de uniforme, y muy pocos de xavier se vieron cortados en las primeras epocas, probablemente debido a una mala iniciacion. En la uniforme es esperable, y aunque xavier esta pensado para evitarlo puede tener ciertos problemas en algunos casos con la ReLU de activacion. 
- ¿Qué impacto tiene la inicialización sobre las métricas de validación?
  Respuesta: Permite una convergencia mas rapida, ya que una mala iniciacion puede trabar totalmente el entrenamiento, o generar una perdida de tiempo en las primeras iteraciones tratando de corregir este desbalance. Ademas una buena inicializacion mantiene la varianza de las activaciones constante, ayudando a impedir el vanishing o exploding gradient. En los ejemplos, aunque vimos que bajo las mismas condiciones en el resto, tanto he como xavier pueden alcanzar los mismos techos, por lo menos en las simulaciones realizadas, he suele converger en menor cantidad de epocas.
- ¿Por qué `bias` se suele inicializar en cero?
  Respuesta: Para no alterar la media igual a cero, justamente la inicializacion busca mantener normalizado 