from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd
df = pd.read_csv("class_german_credit.csv", sep=',')
X = df.drop('Risk', axis=1)
Y = df['Risk']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42, stratify=Y)

#Qntd de Duplicatas
print('Total Duplicatas: ', df.duplicated().sum())

#Tratamento de Outliers numéricos usando Capping
colunas_numericas = ['Age', 'Credit amount', 'Duration']
for coluna in colunas_numericas:
    Q1 = X_train[coluna].quantile(0.25)
    Q3 = X_train[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    X_train[coluna] = X_train[coluna].clip(upper=limite_superior)
    X_test[coluna] = X_test[coluna].clip(upper=limite_superior)
#Tratamento de Outliers numéricos usando agrupamento por categorias raras
colunas_categoricas = ['Purpose', 'Sex', 'Housing', 'Saving accounts', 'Checking account', 'Job']
for coluna in colunas_categoricas:
    freq = X_train[coluna].value_counts(normalize = True)
    categorias_raras = freq[freq<0.05].index
    X_train[coluna] = X_train[coluna].replace(categorias_raras, 'Other')
    X_test[coluna] = X_test[coluna].replace(categorias_raras, 'Other')

#Preenchimento dos missing values com unknown
colunas_para_preencher = ['Saving accounts', 'Checking account']
for coluna in colunas_para_preencher:
    X_train[coluna] = X_train[coluna].fillna('unknown')
    X_test[coluna] = X_test[coluna].fillna('unknown')


X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

dt = DecisionTreeClassifier(class_weight= 'balanced', random_state=42)
dt.fit(X_train, Y_train)
Y_pred = dt.predict(X_test)
print('Acuracia: ', accuracy_score(Y_test, Y_pred))
print('\nMatriz Confus: ', )
print(confusion_matrix(Y_test, Y_pred))
print('\nRelatorio de Classificação: ')
print(classification_report(Y_test, Y_pred))

