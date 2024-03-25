import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def cargar_y_preparar_datos(ruta_malignos, ruta_benignos):
    # Cargar los datos
    malignos = pd.read_csv(ruta_malignos, header=None)
    benignos = pd.read_csv(ruta_benignos, header=None)
    
    # Asignar etiquetas: 1 para malignos, 0 para benignos
    malignos['Etiqueta'] = 1
    benignos['Etiqueta'] = 0
    
    # Combinar los conjuntos de datos
    datos = pd.concat([malignos, benignos], axis=0)
    
    # Mezclar los datos
    datos = datos.sample(frac=1).reset_index(drop=True)
    
    return datos

def dividir_datos(datos):
    X = datos.drop('Etiqueta', axis=1)
    y = datos['Etiqueta']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    return X_train, X_test, y_train, y_test

def entrenar_modelo(X_train, y_train):
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    return modelo

def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nMatriz de Confusión:\n", confusion_matrix(y_test, y_pred))
    print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))
    
    # Opcional: Mostrar la importancia de las características
    plt.figure(figsize=(10, 6))
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]
    plt.title('Importancia de las Características')
    plt.bar(range(X_test.shape[1]), importancias[indices], align='center')
    plt.xticks(range(X_test.shape[1]), indices)
    plt.xlim([-1, X_test.shape[1]])
    plt.show()

def igualar_filas_dataframe_aleatoriamente(ruta_df_grande, ruta_df_pequeno):
    # Cargar ambos DataFrames
    df_grande = pd.read_csv(ruta_df_grande)
    df_pequeno = pd.read_csv(ruta_df_pequeno)
 # Cargar ambos DataFrames
    df_grande = pd.read_csv(ruta_df_grande)
    df_pequeno = pd.read_csv(ruta_df_pequeno)

    # Determinar si el DataFrame grande necesita ser reducido
    if len(df_grande) > len(df_pequeno):
        # Reducir el DataFrame grande para que tenga el mismo número de filas que el pequeño
        df_grande_reducido = df_grande.sample(n=len(df_pequeno), random_state=42)
        # Guardar el DataFrame reducido a un nuevo archivo CSV
        # Asegúrate de cambiar la ruta o el nombre del archivo según necesites
        ruta_nueva_csv = ruta_df_grande.replace('.csv', 'Igualado.csv')
        df_grande_reducido.to_csv(ruta_nueva_csv, index=False)
        print(f"Archivo reducido guardado en: {ruta_nueva_csv}")
        return ruta_nueva_csv
    else:
        # Si el DataFrame grande ya tiene el mismo número de filas o menos, no hacer nada
        print("No se necesitan ajustes, el archivo grande no tiene más filas que el pequeño.")
        return ruta_df_grande


 
# Asumiendo que los datos están en 'datos_malignos.csv' y 'datos_benignos.csv'
ruta_malignos = 'MalignosMultiplicados.csv'
ruta_benignos = 'BenignosMultiplicados.csv'
# Igualar el número de filas
# Ejemplo de uso


datos = cargar_y_preparar_datos(ruta_malignos, ruta_benignos)

X_train, X_test, y_train, y_test = dividir_datos(datos)
modelo = entrenar_modelo(X_train, y_train)
evaluar_modelo(modelo, X_test, y_test)

#-------------------------------------------------------

# Ejemplo de una nueva sentencia
nueva_sentencia = [0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  # Asegúrate de que tenga la misma longitud que las características del modelo

# Convertir en DataFrame de Pandas para una única observación
nueva_sentencia_df = pd.DataFrame([nueva_sentencia])

def predecir_nueva_sentencia(modelo, nueva_sentencia_df):
    pred = modelo.predict(nueva_sentencia_df)
    return pred

# Llamando a la función de predicción
prediccion = predecir_nueva_sentencia(modelo, nueva_sentencia_df)

if prediccion == 1:
    print("La sentencia es maligna.")
else:
    print("La sentencia es benigna.")

def cargar_datos_para_prediccion(ruta_archivo):

    datos = pd.read_csv(ruta_archivo, header=None)
 
    
    return datos


def contar_cambios_prediccion_global(X, modelo):
    """
    Analiza el impacto de cambiar cada característica de 0 a 1 o de 1 a 0 en las predicciones del modelo
    para todo el conjunto de datos X.

    :param X: DataFrame de Pandas con las características de las muestras.
    :param modelo: Modelo de clasificación entrenado.
    :return: Vector con el contador de cambios para cada característica.
    """
    num_caracteristicas = X.shape[1]
    cambios_por_caracteristica = np.zeros(num_caracteristicas)
    
    # Predicciones originales para todo el conjunto de datos
    predicciones_originales = modelo.predict(X)
    
    # Iterar sobre cada característica
    for j in range(num_caracteristicas):
        # Copiar el DataFrame para no modificar el original
        X_modificado = X.copy()
        # Cambiar la característica j-ésima de todas las muestras
        X_modificado.iloc[:, j] = 1 - X_modificado.iloc[:, j]
        # Evaluar predicción con la característica modificada para todo el conjunto
        predicciones_modificadas = modelo.predict(X_modificado)
        # Contar cuántas veces la predicción cambia para esta característica
        cambios = np.sum(predicciones_modificadas != predicciones_originales)
        cambios_por_caracteristica[j] = cambios
    
    return cambios_por_caracteristica

def contar_cambios_prediccion_2_caracteristicas_optimizado(X, modelo):
    num_caracteristicas = X.shape[1]
    cambios_resultantes = []

    for j in range(num_caracteristicas - 1):
        for k in range(j + 1, num_caracteristicas):
            print(f"Analizando características {j} y {k}...")
            X_modificado = X.copy()
            X_modificado.iloc[:, [j, k]] = 1 - X_modificado.iloc[:, [j, k]]
            predicciones_modificadas = modelo.predict(X_modificado)
            cambios = np.sum(predicciones_modificadas != modelo.predict(X))
            cambios_resultantes.append([j, k, cambios])

    # Convertir resultados a DataFrame
    df_cambios = pd.DataFrame(cambios_resultantes, columns=['Característica 1', 'Característica 2', 'Cambios'])

    return df_cambios

def guardar_cambios_en_csv(cambios_resultantes, nombre_archivo="cambios_predicciones_completas.csv"):
    df_cambios = pd.DataFrame(cambios_resultantes, columns=['Característica 1', 'Característica 2', 'Cambios'])
    df_cambios.to_csv(nombre_archivo, index=False)
    
def graficar_caracteristicas_importantes(cambios_globales):
    # Ordenar los cambios por importancia y obtener los 10 más altos
    indices_importantes = np.argsort(cambios_globales)[::-1][:10]
    cambios_importantes = cambios_globales[indices_importantes]
    
    # Calcular el total de cambios para todos los cambios globales
    total_cambios = np.sum(cambios_globales)
    
    # Calcular el porcentaje de cada cambio importante del total de todos los cambios
    porcentajes_cambios = (cambios_importantes / total_cambios) * 100
    
    # Crear las etiquetas para las barras
    etiquetas = [f"Característica {i+1}" for i in indices_importantes]
    
    # Crear la gráfica de barras
    plt.figure(figsize=(10, 6))
    barras = plt.bar(etiquetas, porcentajes_cambios, color='skyblue')
    
    # Configurar título y etiquetas con tamaño de fuente más pequeño
    plt.title('Top 10 Características que más cambian la predicción (%)', fontsize=10)
    plt.xlabel('Características', fontsize=7)
    plt.ylabel('Porcentaje del cambio total', fontsize=9)
    
    # Ajustar las etiquetas del eje X
    plt.xticks(fontsize=8)
    
    # Colocar un texto sobre cada barra con el porcentaje
    for barra in barras:
        altura = barra.get_height()
        plt.text(barra.get_x() + barra.get_width() / 2., altura + 0.5,
                 f'{altura:.2f}%', ha='center', va='bottom', fontsize=8)

    # Mostrar la gráfica
    plt.tight_layout()
    plt.show()
    
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def cargar_y_preparar_datos(ruta_malignos, ruta_benignos):
    # Cargar los datos
    malignos = pd.read_csv(ruta_malignos, header=None)
    benignos = pd.read_csv(ruta_benignos, header=None)
    
    # Asignar etiquetas: 1 para malignos, 0 para benignos
    malignos['Etiqueta'] = 1
    benignos['Etiqueta'] = 0
    
    # Combinar los conjuntos de datos
    datos = pd.concat([malignos, benignos], axis=0)
    
    # Mezclar los datos
    datos = datos.sample(frac=1).reset_index(drop=True)
    
    return datos

def dividir_datos(datos):
    X = datos.drop('Etiqueta', axis=1)
    y = datos['Etiqueta']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    return X_train, X_test, y_train, y_test

def entrenar_modelo(X_train, y_train):
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    return modelo

def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nMatriz de Confusión:\n", confusion_matrix(y_test, y_pred))
    print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))
    
    # Opcional: Mostrar la importancia de las características
    plt.figure(figsize=(10, 6))
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]
    plt.title('Importancia de las Características')
    plt.bar(range(X_test.shape[1]), importancias[indices], align='center')
    plt.xticks(range(X_test.shape[1]), indices)
    plt.xlim([-1, X_test.shape[1]])
    plt.show()


# Asumiendo que los datos están en 'datos_malignos.csv' y 'datos_benignos.csv'
ruta_malignos = 'MalignosMultiplicados.csv'
ruta_benignos = 'BenignosMultiplicados.csv'


datos = cargar_y_preparar_datos(ruta_malignos, ruta_benignos)

X_train, X_test, y_train, y_test = dividir_datos(datos)
modelo = entrenar_modelo(X_train, y_train)
evaluar_modelo(modelo, X_test, y_test)

#-------------------------------------------------------

# Ejemplo de una nueva sentencia
nueva_sentencia = [0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  # Asegúrate de que tenga la misma longitud que las características del modelo

# Convertir en DataFrame de Pandas para una única observación
nueva_sentencia_df = pd.DataFrame([nueva_sentencia])

def predecir_nueva_sentencia(modelo, nueva_sentencia_df):
    pred = modelo.predict(nueva_sentencia_df)
    return pred

# Llamando a la función de predicción
prediccion = predecir_nueva_sentencia(modelo, nueva_sentencia_df)

if prediccion == 1:
    print("La sentencia es maligna.")
else:
    print("La sentencia es benigna.")

def cargar_datos_para_prediccion(ruta_archivo):

    datos = pd.read_csv(ruta_archivo, header=None)
 
    
    return datos


def contar_cambios_prediccion_global(X, modelo):
    """
    Analiza el impacto de cambiar cada característica de 0 a 1 o de 1 a 0 en las predicciones del modelo
    para todo el conjunto de datos X.

    :param X: DataFrame de Pandas con las características de las muestras.
    :param modelo: Modelo de clasificación entrenado.
    :return: Vector con el contador de cambios para cada característica.
    """
    num_caracteristicas = X.shape[1]
    cambios_por_caracteristica = np.zeros(num_caracteristicas)
    
    # Predicciones originales para todo el conjunto de datos
    predicciones_originales = modelo.predict(X)
    
    # Iterar sobre cada característica
    for j in range(num_caracteristicas):
        # Copiar el DataFrame para no modificar el original
        X_modificado = X.copy()
        # Cambiar la característica j-ésima de todas las muestras
        X_modificado.iloc[:, j] = 1 - X_modificado.iloc[:, j]
        # Evaluar predicción con la característica modificada para todo el conjunto
        predicciones_modificadas = modelo.predict(X_modificado)
        # Contar cuántas veces la predicción cambia para esta característica
        cambios = np.sum(predicciones_modificadas != predicciones_originales)
        cambios_por_caracteristica[j] = cambios
    
    return cambios_por_caracteristica

def contar_cambios_prediccion_2_caracteristicas_optimizado(X, modelo):
    num_caracteristicas = X.shape[1]
    cambios_resultantes = []

    for j in range(num_caracteristicas - 1):
        for k in range(j + 1, num_caracteristicas):
            print(f"Analizando características {j} y {k}...")
            X_modificado = X.copy()
            X_modificado.iloc[:, [j, k]] = 1 - X_modificado.iloc[:, [j, k]]
            predicciones_modificadas = modelo.predict(X_modificado)
            cambios = np.sum(predicciones_modificadas != modelo.predict(X))
            cambios_resultantes.append([j, k, cambios])

    # Convertir resultados a DataFrame
    df_cambios = pd.DataFrame(cambios_resultantes, columns=['Característica 1', 'Característica 2', 'Cambios'])

    return df_cambios

def guardar_cambios_en_csv(cambios_resultantes, nombre_archivo="cambios_predicciones_completas.csv"):
    df_cambios = pd.DataFrame(cambios_resultantes, columns=['Característica 1', 'Característica 2', 'Cambios'])
    df_cambios.to_csv(nombre_archivo, index=False)
    
def graficar_caracteristicas_importantes(cambios_globales):
    # Ordenar los cambios por importancia y obtener los 10 más altos
    indices_importantes = np.argsort(cambios_globales)[::-1][:10]
    cambios_importantes = cambios_globales[indices_importantes]
    
    # Calcular el total de cambios para todos los cambios globales
    total_cambios = np.sum(cambios_globales)
    
    # Calcular el porcentaje de cada cambio importante del total de todos los cambios
    porcentajes_cambios = (cambios_importantes / total_cambios) * 100
    
    # Crear las etiquetas para las barras
    etiquetas = [f"Característica {i+1}" for i in indices_importantes]
    
    # Crear la gráfica de barras
    plt.figure(figsize=(10, 6))
    barras = plt.bar(etiquetas, porcentajes_cambios, color='skyblue')
    
    # Configurar título y etiquetas con tamaño de fuente más pequeño
    plt.title('Top 10 Características que más cambian la predicción (%)', fontsize=10)
    plt.xlabel('Características', fontsize=7)
    plt.ylabel('Porcentaje del cambio total', fontsize=9)
    
    # Ajustar las etiquetas del eje X
    plt.xticks(fontsize=8)
    
    # Colocar un texto sobre cada barra con el porcentaje
    for barra in barras:
        altura = barra.get_height()
        plt.text(barra.get_x() + barra.get_width() / 2., altura + 0.5,
                 f'{altura:.2f}%', ha='center', va='bottom', fontsize=8)

    # Mostrar la gráfica
    plt.tight_layout()
    plt.show()
    
def evaluar_cambio_individual_por_muestra_optimizado(X, modelo):
    num_muestras, num_caracteristicas = X.shape
    resultados_cambios = np.zeros((num_muestras, num_caracteristicas))
    predicciones_originales = modelo.predict(X)

    for j in range(num_caracteristicas):
        X_modificada = X.copy()
        X_modificada.iloc[:, j] = 1 - X_modificada.iloc[:, j]
        predicciones_modificadas = modelo.predict(X_modificada)
        cambios = predicciones_modificadas != predicciones_originales
        resultados_cambios[:, j] = cambios

    muestras_con_cambio = np.any(resultados_cambios, axis=1).sum()
    print(f'Muestras que cambiaron su predicción: {muestras_con_cambio}')
    return resultados_cambios

def contar_cambios_prediccion_por_pares_de_caracteristicas(X, modelo):
    num_muestras = X.shape[0]
    num_caracteristicas = X.shape[1]
    contador_cambios = 0

    predicciones_originales = modelo.predict(X)
    X_np = X.values  # Conversión para mejorar el rendimiento

    for i in range(num_muestras):
        cambio_detectado = False  # Bandera para detectar el primer cambio
        for j in range(num_caracteristicas - 1):
            if cambio_detectado:
                break  # Si ya se encontró un cambio, continuar con la siguiente muestra
            for k in range(j + 1, num_caracteristicas):
                # Solo proceder si aún no se ha detectado un cambio para esta muestra
                if not cambio_detectado:
                    fila_modificada = X_np[i, :].copy()
                    # Invertir los valores de las características j y k
                    fila_modificada[[j, k]] = 1 - fila_modificada[[j, k]]

                    prediccion_modificada = modelo.predict(fila_modificada.reshape(1, -1))
                    if prediccion_modificada != predicciones_originales[i]:
                        contador_cambios += 1
                        cambio_detectado = True  # Marcar que se detectó un cambio y no revisar más pares para esta muestra
                        break  # Salir del bucle interno y no evaluar más pares para esta muestra

    return contador_cambios
# Asumiendo que ya has cargado tus datos y entrenado tu modelo como anteriormente.
datos = cargar_y_preparar_datos(ruta_malignos, ruta_benignos)
X_train, X_test, y_train, y_test = dividir_datos(datos)
modelo = entrenar_modelo(X_train, y_train)
evaluar_modelo(modelo, X_test, y_test)
# Corrección en la llamada de la función
datos_para_prediccion = cargar_datos_para_prediccion(ruta_benignos)  # Asumiendo que esta función carga los datos correctamente
cambios_individuales_benignos = evaluar_cambio_individual_por_muestra_optimizado(datos_para_prediccion, modelo)
print(cambios_individuales_benignos)

datos_para_prediccion = cargar_datos_para_prediccion(ruta_malignos)  # Asumiendo que esta función carga los datos correctamente
cambios_individuales_malignos = evaluar_cambio_individual_por_muestra_optimizado(datos_para_prediccion, modelo)
print(cambios_individuales_malignos)
"""""
sumas_cambios_benignos = np.sum(cambios_individuales_benignos, axis=0)
sumas_cambios_malignos = np.sum(cambios_individuales_malignos, axis=0)

# Calcular el total de cambios para cada característica combinando ambos conjuntos
total_cambios_por_caracteristica = sumas_cambios_benignos + sumas_cambios_malignos

# Calcular el porcentaje de cambios para cada conjunto por característica
porcentajes_benignos = (sumas_cambios_benignos / total_cambios_por_caracteristica) * 100
porcentajes_malignos = (sumas_cambios_malignos / total_cambios_por_caracteristica) * 100

# Preparar la gráfica
plt.figure(figsize=(20, 10))

# Nombres de las características
caracteristicas = [f"Característica {i+1}" for i in range(porcentajes_benignos.shape[0])]

# Posiciones de las barras en el eje X
x = np.arange(len(caracteristicas))

# Graficar los porcentajes de cambios para cada conjunto
plt.bar(x - 0.2, porcentajes_benignos, 0.4, label='Benignos')
plt.bar(x + 0.2, porcentajes_malignos, 0.4, label='Malignos')

# Añadir título y etiquetas
plt.ylabel('Porcentaje de cambios')
plt.title('Comparación del porcentaje de cambios por característica entre conjuntos benignos y malignos')
plt.xticks(x, caracteristicas, rotation='vertical')
plt.legend()

# Añadir líneas de cuadrícula para mejorar la legibilidad
plt.grid(axis='y')

# Mostrar la gráfica
plt.tight_layout()
plt.show()"""

# Obtener los totales de cambios para cada tipo de muestra
total_cambios_benignos = np.any(cambios_individuales_benignos, axis=1).sum()
total_cambios_malignos = np.any(cambios_individuales_malignos, axis=1).sum()

# Calcular los porcentajes de cambio con respecto al total de muestras para cada tipo de muestra
porcentaje_cambios_benignos_respecto_total = (total_cambios_benignos / 3246) * 100
porcentaje_cambios_malignos_respecto_total = (total_cambios_malignos / 3246) * 100

# Crear la gráfica de barras ajustada asumiendo que los porcentajes son escalares
labels_respecto_total = ['Benignos', 'Malignos']
porcentajes_respecto_total = [porcentaje_cambios_benignos_respecto_total, porcentaje_cambios_malignos_respecto_total]

fig_ajustada, ax_ajustada = plt.subplots()
bars_ajustada = ax_ajustada.bar(labels_respecto_total, porcentajes_respecto_total, color=['skyblue', 'salmon'])

# Añadir el porcentaje sobre cada barra
ax_ajustada.bar_label(bars_ajustada, labels=[f'{p:.2f}%' for p in porcentajes_respecto_total], padding=3)

# Configurar título y etiquetas
ax_ajustada.set_ylabel('Porcentaje respecto al total de muestras (3246)')
ax_ajustada.set_title('Porcentaje de Cambios en Predicciones por Tipo de Muestra (Respecto al total)')
ax_ajustada.set_ylim(0, 100)  # Establecer el límite superior del eje y en 100%

plt.tight_layout()
plt.show()

# Corrección en la llamada de la función
datos_para_prediccion = cargar_datos_para_prediccion(ruta_benignos)  # Asumiendo que esta función carga los datos correctamente
cambios_por_pares_benignos = contar_cambios_prediccion_por_pares_de_caracteristicas(datos_para_prediccion, modelo)
print(cambios_por_pares_benignos)

datos_para_prediccion = cargar_datos_para_prediccion(ruta_malignos)  # Asumiendo que esta función carga los datos correctamente
cambios_por_pares_malignos = contar_cambios_prediccion_por_pares_de_caracteristicas(datos_para_prediccion, modelo)
print(cambios_por_pares_malignos)

# Calcular porcentajes de cambios
porcentaje_cambios_individuales_benignos = np.mean(cambios_individuales_benignos) * 100
porcentaje_cambios_individuales_malignos = np.mean(cambios_individuales_malignos) * 100
porcentaje_cambios_por_pares_benignos = (cambios_por_pares_benignos / len(ruta_benignos)) * 100
porcentaje_cambios_por_pares_malignos = (cambios_por_pares_malignos / len(ruta_malignos)) * 100

# Preparar datos para la gráfica
categorias = ['Individuales Benignos', 'Individuales Malignos', 'Pares Benignos', 'Pares Malignos']
porcentajes = [
    porcentaje_cambios_individuales_benignos,
    porcentaje_cambios_individuales_malignos,
    porcentaje_cambios_por_pares_benignos,
    porcentaje_cambios_por_pares_malignos
]

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.bar(categorias, porcentajes, color=['blue', 'red', 'green', 'orange'])
plt.xlabel('Tipo de Análisis')
plt.ylabel('Porcentaje de Cambios en Predicciones')
plt.title('Porcentaje de Cambios en Predicciones por Tipo de Análisis')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show() 