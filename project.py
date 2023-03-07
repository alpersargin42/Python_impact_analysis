import numpy as np   #todo Lineer Cebir
import pandas as pd  #! Veri işleme,CSV dosyası Input/Output
import os
import csv


# Veri setinin yüklenmesi
def loadCsv(filename):
    lines = csv.reader(open('student_prediction.csv'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def main():
    filename = 'student_prediction.csv'
    df = pd.read_csv(filename)
    print(df.head())
# main()

