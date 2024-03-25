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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
    
    plt.figure(figsize=(10, 6))
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]
    plt.title('Importancia de las Características')
    plt.bar(range(X_test.shape[1]), importancias[indices], align='center')
    plt.xticks(range(X_test.shape[1]), indices)
    plt.xlim([-1, X_test.shape[1]])
    plt.show()
    

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
    num_caracteristicas = X.shape[1]
    cambios_por_caracteristica = np.zeros(num_caracteristicas)
    
  
    predicciones_originales = modelo.predict(X)
    
   
    for j in range(num_caracteristicas):
        X_modificado = X.copy()
        X_modificado.iloc[:, j] = 1 - X_modificado.iloc[:, j]
        predicciones_modificadas = modelo.predict(X_modificado)
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


    df_cambios = pd.DataFrame(cambios_resultantes, columns=['Característica 1', 'Característica 2', 'Cambios'])

    return df_cambios

def guardar_cambios_en_csv(cambios_resultantes, nombre_archivo="cambios_predicciones_completas.csv"):
    df_cambios = pd.DataFrame(cambios_resultantes, columns=['Característica 1', 'Característica 2', 'Cambios'])
    df_cambios.to_csv(nombre_archivo, index=False)
    
def graficar_caracteristicas_importantes(cambios_globales):
    indices_importantes = np.argsort(cambios_globales)[::-1][:10]
    cambios_importantes = cambios_globales[indices_importantes]

    total_cambios = np.sum(cambios_globales)
    
    porcentajes_cambios = (cambios_importantes / total_cambios) * 100
    
    etiquetas = [f"Característica {i+1}" for i in indices_importantes]
    
    plt.figure(figsize=(10, 6))
    barras = plt.bar(etiquetas, porcentajes_cambios, color='skyblue')
    
    plt.title('Top 10 Características que más cambian la predicción (%)', fontsize=10)
    plt.xlabel('Características', fontsize=7)
    plt.ylabel('Porcentaje del cambio total', fontsize=9)
  
    plt.xticks(fontsize=8)
    
    for barra in barras:
        altura = barra.get_height()
        plt.text(barra.get_x() + barra.get_width() / 2., altura + 0.5,
                 f'{altura:.2f}%', ha='center', va='bottom', fontsize=8)

    # Mostrar la gráfica
    plt.tight_layout()
    plt.show()
    

def contar_cambios_prediccion_detalle(X, modelo):
    """
    Cuenta los cambios en las predicciones del modelo cuando cada característica
    se activa (cambia de 0 a 1) y se desactiva (cambia de 1 a 0) por separado.
    
    :param X: DataFrame de Pandas con las características de las muestras.
    :param modelo: Modelo de clasificación entrenado.
    :return: Dos vectores con los contadores de cambios para cada característica,
             uno para cambios de 0 a 1 y otro para cambios de 1 a 0.
    """
    num_caracteristicas = X.shape[1]
    cambios_a_activado = np.zeros(num_caracteristicas)
    cambios_a_desactivado = np.zeros(num_caracteristicas)
    
    predicciones_originales = modelo.predict(X)
    
    for caracteristica in range(num_caracteristicas):
        # Activar la característica si estaba desactivada
        X_activado = X.copy()
        X_activado.iloc[:, caracteristica] = X_activado.iloc[:, caracteristica].apply(lambda x: 1 if x == 0 else x)
        predicciones_activado = modelo.predict(X_activado)
        cambios_a_activado[caracteristica] = np.sum(predicciones_activado != predicciones_originales)
        
        # Desactivar la característica si estaba activada
        X_desactivado = X.copy()
        X_desactivado.iloc[:, caracteristica] = X_desactivado.iloc[:, caracteristica].apply(lambda x: 0 if x == 1 else x)
        predicciones_desactivado = modelo.predict(X_desactivado)
        cambios_a_desactivado[caracteristica] = np.sum(predicciones_desactivado != predicciones_originales)
    
    return cambios_a_activado, cambios_a_desactivado

def imprimir_cambios_como_enteros(cambios):
    cambios_enteros = [int(cambio) for cambio in cambios]  # Convertir cada elemento a entero
    print(cambios_enteros)
datos_para_prediccion = cargar_datos_para_prediccion(ruta_benignos)
cambios_globales = contar_cambios_prediccion_global(datos_para_prediccion, modelo)
graficar_caracteristicas_importantes(cambios_globales)
vector_contador_cambio = []

print("Cambios globales por característica:")
for i, cambio in enumerate(cambios_globales):
    print(f"Característica {i}: Cambios = {cambio}")
    vector_contador_cambio.append(cambio)
    
print(vector_contador_cambio)

datos_para_prediccion = cargar_datos_para_prediccion(ruta_malignos)
cambios_globales = contar_cambios_prediccion_global(datos_para_prediccion, modelo)
graficar_caracteristicas_importantes(cambios_globales)
vector_contador_cambio = []
# Imprimir los cambios para cada característica
print("Cambios por característica Malignas:")
for i, cambio in enumerate(cambios_globales):
    print(f"Característica {i}: Cambios = {cambio}")
    vector_contador_cambio.append(cambio)
    
print(vector_contador_cambio)

datos_malignos = datos[datos['Etiqueta'] == 1].drop('Etiqueta', axis=1)


datos_benignos = datos[datos['Etiqueta'] == 0].drop('Etiqueta', axis=1)

cambios_a_activado_malignos, cambios_a_desactivado_malignos = contar_cambios_prediccion_detalle(datos_malignos, modelo)

cambios_a_activado_benignos, cambios_a_desactivado_benignos = contar_cambios_prediccion_detalle(datos_benignos, modelo)
print("Cambios globales por característica:")
imprimir_cambios_como_enteros(cambios_globales)

print("Malignos - Cambios al activar una característica:")
imprimir_cambios_como_enteros(cambios_a_activado_malignos)

print("Malignos - Cambios al desactivar una característica:")
imprimir_cambios_como_enteros(cambios_a_desactivado_malignos)

print("\nBenignos - Cambios al activar una característica:")
imprimir_cambios_como_enteros(cambios_a_activado_benignos)

print("Benignos - Cambios al desactivar una característica:")
imprimir_cambios_como_enteros(cambios_a_desactivado_benignos)


def calcular_top_5(cambios):
    indices_top_5 = np.argsort(cambios)[::-1][:5] 
    valores_top_5 = np.sort(cambios)[::-1][:5]  
    return indices_top_5, valores_top_5
top_5_activado_malignos_indices, top_5_activado_malignos_valores = calcular_top_5(cambios_a_activado_malignos)
top_5_desactivado_malignos_indices, top_5_desactivado_malignos_valores = calcular_top_5(cambios_a_desactivado_malignos)
top_5_activado_benignos_indices, top_5_activado_benignos_valores = calcular_top_5(cambios_a_activado_benignos)
top_5_desactivado_benignos_indices, top_5_desactivado_benignos_valores = calcular_top_5(cambios_a_desactivado_benignos)

(top_5_activado_malignos_indices, top_5_activado_malignos_valores, 
 top_5_desactivado_malignos_indices, top_5_desactivado_malignos_valores, 
 top_5_activado_benignos_indices, top_5_activado_benignos_valores, 
 top_5_desactivado_benignos_indices, top_5_desactivado_benignos_valores)

def graficar_top_5(indices, valores, titulo, total_cambios):
    """
    Función para graficar los top 5 cambios, mostrando los porcentajes respecto al total de cambios.
    """
    plt.figure(figsize=(10, 5))
    barras = plt.bar(range(len(valores)), valores, color='skyblue')
    plt.xlabel('Características')
    plt.ylabel('Número de Cambios')
    plt.title(titulo)
    plt.xticks(range(len(valores)), [f'Característica {i}' for i in indices + 1])
    
    for barra in barras:
        height = barra.get_height()
        porcentaje = (height / total_cambios) * 100
        plt.text(barra.get_x() + barra.get_width() / 2., height,
                 f'{porcentaje:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
total_cambios_malignos = sum(cambios_a_activado_malignos) + sum(cambios_a_desactivado_malignos)
total_cambios_benignos = sum(cambios_a_activado_benignos) + sum(cambios_a_desactivado_benignos)

# Graficar los resultados para cada categoría corrigiendo el llamado a la función
graficar_top_5(top_5_activado_malignos_indices, top_5_activado_malignos_valores, 'Top 5 Activados - Malignos', total_cambios_malignos)
graficar_top_5(top_5_desactivado_malignos_indices, top_5_desactivado_malignos_valores, 'Top 5 Desactivados - Malignos', total_cambios_malignos)
graficar_top_5(top_5_activado_benignos_indices, top_5_activado_benignos_valores, 'Top 5 Activados - Benignos', total_cambios_benignos)
graficar_top_5(top_5_desactivado_benignos_indices, top_5_desactivado_benignos_valores, 'Top 5 Desactivados - Benignos', total_cambios_benignos)

def buscar_sentencia_en_dataset(sentencia, dataset):
    """
    Busca si una sentencia modificada existe en otro dataset.
    
    :param sentencia: La sentencia modificada como un DataFrame de Pandas.
    :param dataset: El DataFrame de Pandas del otro dataset donde buscar.
    :return: True si la sentencia existe en el dataset, False en caso contrario.
    """
    existe = any((dataset.values == sentencia.values).all(1))
    return existe

def evaluar_cambios_entre_datasets(X_malignos, X_benignos, modelo):
    datasets = {'maligno': X_malignos, 'benigno': X_benignos}
    for nombre_origen, X_origen in datasets.items():
        nombre_destino = 'benigno' if nombre_origen == 'maligno' else 'maligno'
        X_destino = datasets[nombre_destino]
        
        for i, sentencia in X_origen.iterrows():
            for caracteristica in range(X_origen.shape[1]):
                sentencia_modificada = sentencia.copy()
                sentencia_modificada.iloc[caracteristica] = 1 - sentencia_modificada.iloc[caracteristica]
            
                sentencia_modificada_df = pd.DataFrame([sentencia_modificada])
                
                prediccion_original = modelo.predict(sentencia.values.reshape(1, -1))[0]
                prediccion_modificada = modelo.predict(sentencia_modificada_df)[0]
                
                if prediccion_original != prediccion_modificada:
                    if buscar_sentencia_en_dataset(sentencia_modificada_df, X_destino):
                        print(f"Sentencia modificada de {nombre_origen} a {nombre_destino}, encontrada en dataset {nombre_destino}. Feature modificada: {caracteristica + 1}")

                        print(f"Sentencia: {sentencia_modificada.values}")

X_malignos = datos[datos['Etiqueta'] == 1].drop('Etiqueta', axis=1)
X_benignos = datos[datos['Etiqueta'] == 0].drop('Etiqueta', axis=1)
evaluar_cambios_entre_datasets(X_malignos, X_benignos, modelo)
