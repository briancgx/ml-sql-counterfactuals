import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

def cargar_y_preparar_datos(ruta_malignos, ruta_benignos):
    # Cargar los datos sin encabezado y asignar nombres de columnas como strings
    column_names = [str(i) for i in range(len(pd.read_csv(ruta_malignos, nrows=1).columns))]
    malignos = pd.read_csv(ruta_malignos, header=None, names=column_names)
    benignos = pd.read_csv(ruta_benignos, header=None, names=column_names)
    
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
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

# Rutas a los archivos CSV
ruta_malignos = 'MalignosMultiplicados.csv'
ruta_benignos = 'BenignosMultiplicados.csv'

# Cargar, preparar y dividir los datos
datos = cargar_y_preparar_datos(ruta_malignos, ruta_benignos)
X_train, X_test, y_train, y_test = dividir_datos(datos)

# Entrenar el modelo y evaluarlo
modelo = entrenar_modelo(X_train, y_train)
evaluar_modelo(modelo, X_test, y_test)

# Crear el objeto explainer
explainer = ClassifierExplainer(modelo, X_test, y_test)

# Iniciar el dashboard
db = ExplainerDashboard(explainer, title="Mi Dashboard de Modelo", port=8051)
db.run()
