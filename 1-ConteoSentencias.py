import pandas as pd

# Carga el dataset
df = pd.read_csv('benignos.csv', header=None)

# Convierte cada fila a una tupla (para que sea hashable y pueda ser contada)
tuples = [tuple(x) for x in df.values]

# Cuenta la frecuencia de cada vector único
frequencies = pd.Series(tuples).value_counts()

# Convierte las frecuencias a un DataFrame para guardarlo fácilmente
frequencies_df = frequencies.reset_index()
frequencies_df.columns = ['Vector', 'Frequency']

# Guarda el nuevo dataset en un archivo CSV
frequencies_df.to_csv('frequencies_benignos.csv', index=False, header=None)

print("El archivo 'frequencies_benignos.csv' ha sido guardado con éxito.")
