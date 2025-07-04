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

#Preenchimento missing values KNN
# Supondo que X_train, X_test, Y_train, Y_test já existem

# 1. Crie cópias para evitar avisos
X_train_knn = X_train.copy()
X_test_knn = X_test.copy()

# 2. Mapeie manualmente as colunas ordinais com dados faltantes para números
# Isso converte as categorias para números, mas mantém NaN como NaN
mapa_contas = {'little': 1, 'moderate': 2, 'rich': 3, 'quite rich': 4}
X_train_knn['Saving accounts'] = X_train_knn['Saving accounts'].map(mapa_contas)
X_train_knn['Checking account'] = X_train_knn['Checking account'].map(mapa_contas)
X_test_knn['Saving accounts'] = X_test_knn['Saving accounts'].map(mapa_contas)
X_test_knn['Checking account'] = X_test_knn['Checking account'].map(mapa_contas)

# 3. Aplique get_dummies nas outras colunas categóricas que não têm faltantes
X_train_knn = pd.get_dummies(X_train_knn, drop_first=True)
X_test_knn = pd.get_dummies(X_test_knn, drop_first=True)

# 4. Alinhe as colunas após o dummies
X_train_knn, X_test_knn = X_train_knn.align(X_test_knn, join='left', axis=1, fill_value=0)

# 5. AGORA, com tudo numérico (exceto os NaNs), use o KNNImputer

imputer = KNNImputer(n_neighbors=5)

# Aprenda com o treino e transforme ambos
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_knn), columns=X_train_knn.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test_knn), columns=X_test_knn.columns)

# 6. O resultado (X_train_imputed) está pronto para ser usado no seu modelo
# dt.fit(X_train_imputed, Y_train)
# ...
'''
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
'''
dt = DecisionTreeClassifier(class_weight= 'balanced', random_state=42)
dt.fit(X_train_imputed, Y_train)
Y_pred = dt.predict(X_test_imputed)
print('Acuracia: ', accuracy_score(Y_test, Y_pred))
print('\nMatriz Confus: ', )
print(confusion_matrix(Y_test, Y_pred))
print('\nRelatorio de Classificação: ')
print(classification_report(Y_test, Y_pred))

