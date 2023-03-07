import numpy as np   #todo Lineer Cebir
import pandas as pd  #! Veri işleme,CSV dosyası Input/Output
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from Session_Control2 import *
from jinja2 import *
from openpyxl import *
from sklearn.feature_selection import mutual_info_regression

# Veri setinin yüklenmesi
def loadCsv(filename):
    lines = csv.reader(open('student_prediction.csv'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def main_graph2():
    filename = 'student_prediction.csv'
    df = pd.read_csv(filename)
    print(df.shape)
    oranges=df.describe().T.style.background_gradient(cmap="Oranges")
    if session_control6()==False:
        oranges.to_excel("Oranges.xlsx")
    df["COURSE ID"].unique()
    print(df["COURSE ID"].unique())
    print("\n")
    df.describe(include=object)
    print(df.describe(include=object))
    print("\n")
    df = df.drop('STUDENTID', axis=1)
    duplicate = df[df.duplicated()]
    print("Duplicate Rows :",duplicate)

    sns.countplot(x=df['GRADE'], label="Count")
    if session_control7() == False:
        plt.savefig('etki_sayac.png')
    plt.show()
    
    def make_mi_scores(X, y):
        X = X.copy()
        for colname in X.select_dtypes(["object", "category"]):
            X[colname], _ = X[colname].factorize()
        # All discrete features should now have integer dtypes
        discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
        mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores

    X = df.drop('GRADE', axis=1)
    y = df['GRADE']

    mi_scores = make_mi_scores(X, y)
    print(mi_scores.head(20))

    def drop_uninformative(df, mi_scores):
        return df.loc[:, mi_scores > 0]
    X = drop_uninformative(X, mi_scores)

    random_forest = RandomForestClassifier(random_state=0) #todo from sklearn.ensemble import RandomForestClassifier
    random_forest.fit(X, y)
    predict = cross_val_predict(estimator=random_forest, X=X, y=y, cv=5)
    print("\n")
    print("Sınıflandırma Raporu(RandomForest): \n", classification_report(y, predict))

    decision_tree = dtc(random_state=0) #todo from sklearn.tree import DecisionTreeClassifier as dtc
    decision_tree.fit(X, y)
    predict = cross_val_predict(estimator=decision_tree, X=X, y=y, cv=5)
    print("\n")
    print("Sınıflandırma Raporu(DTC): \n", classification_report(y, predict))

    knn = KNeighborsClassifier() #todo from sklearn.neighbors import KNeighborsClassifier
    knn.fit(X, y)
    predict = cross_val_predict(estimator=knn, X=X, y=y, cv=5) #! Scikit-learn temel arayüzüdür. Kısaca, datayı öğrenebilen class’lara estimator denir.
    print("\n") #?sklearn.model_selection.cross_val_score(estimator, X, y=None, *, groups=None, scoring=None, cv=None, n_jobs=None, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', error_score=nan)
    print("Sınıflandırma Raporu(KNN): \n", classification_report(y, predict))

    gnb = GaussianNB() #todo from sklearn.naive_bayes import GaussianNB
    gnb.fit(X, y)
    predict = cross_val_predict(estimator=gnb, X=X, y=y, cv=5) #! Teşhis amaçları için her bir çapraz doğrulama bölümünden tahminler alın.
    print("\n")
    print("Sınıflandırma Raporu(GaussianNB): \n", classification_report(y, predict))

    scv = SVC() #todo from sklearn.svm import SVC
    scv.fit(X, y)
    predict = cross_val_predict(estimator=scv, X=X, y=y, cv=5)
    print("\n")
    print("Sınıflandırma Raporu(SVC): \n", classification_report(y, predict))

# main_graph2()
