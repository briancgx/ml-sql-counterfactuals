import csv

archivo_entrada = 'frequencies_benignos.csv'
archivo_salida = 'DownBenignos.csv'

with open(archivo_entrada, mode='r') as infile, open(archivo_salida, mode='w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        frecuencia = int(row[1])
        # Ajustar la frecuencia si es mayor a 100
        if frecuencia > 100:
            frecuencia = 100
        # Escribir la fila ajustada en el archivo de salida
        writer.writerow([row[0], frecuencia])

print(f'Archivo "{archivo_salida}" generado con éxito.')
