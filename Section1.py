import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
import csv
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from Session_Control1 import *

# Veri setinin yüklenmesi
def loadCsv(filename):
    lines = csv.reader(open('student_prediction.csv'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def main_graph1():
    filename = 'student_prediction.csv'
    df = pd.read_csv(filename)
    X = df.drop(['GRADE', 'STUDENTID'], axis=1)
    y = df['GRADE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    importances = pd.DataFrame(data={'Attribute': X_train.columns, 'Importance': model.coef_[0]})
    importances = importances.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(16, 8))
    plt.bar(x=importances['Attribute'], height=importances['Importance'])
    plt.title('Katsayılardan elde edilen özelliğin önemi', size=20)
    plt.xticks(rotation='vertical')
    if session_control1()==False: 
        plt.savefig('graph1.png')
    plt.show()
    new_df=df[['MOTHER_JOB','FATHER_JOB','SALARY','KIDS','COURSE ID','IMPACT','GRADE']] #todo hangi başlıkların grafiğini istersek buraya yazmalıyız.
    sns.pairplot(new_df,hue_order=['GRADE','IMPACT']) #todo seviye ve etki grafiği.
    if session_control2()==False: 
        plt.savefig('graph2.png')
    plt.show()
    plt.figure(figsize=(14, 5))
    sns.countplot(x=new_df['SALARY'], order=np.arange(1,6,1), hue=new_df['GRADE']) #! ailelerin kazançlarına göre etki oranı.
    plt.xticks(np.arange(0, 5), ['USD 135-200', 'USD 201-270', 'USD 271-340', 'USD 341-410', 'above 410']) #todo para aralığı
    plt.legend(['F', 'DD', 'DC', 'CC', 'CB', 'BB', 'BA', 'AA'], loc=1) #sınıflandırma
    if session_control3()==False: 
        plt.savefig('paraya_gore_etki_sinif.png')
    plt.show()
    plt.figure(figsize=(14, 5))
    sns.countplot(x=new_df['KIDS'], order=np.arange(1, 4), hue=new_df['GRADE'])
    plt.xticks(np.arange(0, 3), ['married', 'divorced', 'died - one of them or both'])
    plt.legend(['F', 'DD', 'DC', 'CC', 'CB', 'BB', 'BA', 'AA'])
    if session_control4()==False: 
        plt.savefig('ailevi_duruma_gore_etki_sinif.png')
    plt.show()
    plt.figure(figsize=(14, 5))
    sns.countplot(x=new_df['FATHER_JOB'], order=np.arange(1, 6), hue=new_df['GRADE'])
    plt.xticks(np.arange(0, 5), ['retired', 'government officer', 'private sector employee', 'self-employment', 'other'])
    plt.legend(['F', 'DD', 'DC', 'CC', 'CB', 'BB', 'BA', 'AA'])
    if session_control5()==False: 
        plt.savefig('babanın_isine_gore_etki_sinif.png')
    plt.show()
# main_graph1()

