import pandas as pd

df = pd.read_csv('DownMalignos.csv', header=None)

def vector_to_list(vector_str):
    return list(map(int, vector_str.strip("()").split(",")))

def generate_rows(vector_str, frequency):
    vector = vector_to_list(vector_str)
    return [vector] * frequency

all_rows = []

for index, row in df.iterrows():
    vector_str, frequency = row[0], row[1]
    all_rows.extend(generate_rows(vector_str, frequency))

new_df = pd.DataFrame(all_rows)

new_df.to_csv('MalignosMultiplicados.csv', header=False, index=False)

print("El nuevo dataset ha sido generado con éxito.")
