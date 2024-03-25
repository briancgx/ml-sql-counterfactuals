import pandas as pd

# Carga el dataset
df = pd.read_csv('benignos.csv', header=None)

tuples = [tuple(x) for x in df.values]

frequencies = pd.Series(tuples).value_counts()

frequencies_df = frequencies.reset_index()
frequencies_df.columns = ['Vector', 'Frequency']

frequencies_df.to_csv('frequencies_benignos.csv', index=False, header=None)

print("El archivo 'frequencies_benignos.csv' ha sido guardado con éxito.")
