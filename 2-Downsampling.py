import csv

# Nombre del archivo de entrada y salida
archivo_entrada = 'frequencies_malignos.csv'
archivo_salida = 'DownMalignos.csv'

# Abrir el archivo de entrada y el archivo de salida
with open(archivo_entrada, mode='r') as infile, open(archivo_salida, mode='w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        # Asumiendo que la frecuencia es el segundo elemento de cada fila
        frecuencia = int(row[1])
        # Ajustar la frecuencia si es mayor a 100
        if frecuencia > 38:
            frecuencia = 38
        # Escribir la fila ajustada en el archivo de salida
        writer.writerow([row[0], frecuencia])

print(f'Archivo "{archivo_salida}" generado con éxito.')
