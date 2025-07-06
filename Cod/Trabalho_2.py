from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree #
from imblearn.over_sampling import SMOTE
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


#Discretização
colunas_numericas_para_discretizar = ['Age', 'Credit amount', 'Duration'] 
num_bins = 4

for coluna in colunas_numericas_para_discretizar:
    # Cria um nome para a nova coluna, ex: 'Age_binned'
    nova_coluna_nome = f'{coluna}_binned'   
    # Cria rótulos dinâmicos para os bins, ex: ['Age_Q1', 'Age_Q2', ...]
    labels = [f'{coluna}_Q{i+1}' for i in range(num_bins)]   
    # Aplica a discretização por quantis no conjunto de treino
    X_train[nova_coluna_nome] = pd.qcut(X_train[coluna], q=num_bins,
                                        labels=labels, duplicates='drop')
    
    # Aplica a mesma discretização no conjunto de teste
    X_test[nova_coluna_nome] = pd.qcut(X_test[coluna], q=num_bins, 
                                       labels=labels, duplicates='drop')

X_train = X_train.drop(columns=colunas_numericas_para_discretizar)
X_test = X_test.drop(columns=colunas_numericas_para_discretizar)


X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

dt_podada = DecisionTreeClassifier(
    max_depth=5,              
    min_samples_leaf=15,
    class_weight='balanced',
    random_state=42
)
dt_podada.fit(X_train, Y_train) 
Y_pred = dt_podada.predict(X_test)

print('Acuracia: ', accuracy_score(Y_test, Y_pred))
print('\nMatriz Confus: ', )
print(confusion_matrix(Y_test, Y_pred))
print('\nRelatorio de Classificação: ')
print(classification_report(Y_test, Y_pred))

