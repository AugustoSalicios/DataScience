from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("class_german_credit.csv", sep=',')
#Verificação quanto a existência de dulicatas
print("Total de duplicatas:", df.duplicated().sum())

'''
#Remoção de Outliers
def detectar_outliers_iqr(df, coluna):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return ~df[coluna].between(limite_inferior, limite_superior)
mask_outliers = pd.Series(False, index=df.index)
for coluna in df.select_dtypes(include=['int64', 'float64']).columns:
    mask_outliers |= detectar_outliers_iqr(df, coluna)
df_sem_outliers = df[~mask_outliers].reset_index(drop=True)
print(f"Total de linhas removidas: {mask_outliers.sum()}")


#Aplicação do KNNimputer para assim preencher os missing values
df_encoded = df.copy()
colunas_cat = ['Saving accounts', 'Checking account', 'Sex', 'Housing', 'Purpose']
label_encoders = {}
for col in colunas_cat:
    le = LabelEncoder()
    df_encoded[col] = df_encoded[col].astype(str)  # evitar erro com NaN
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le
target_col = df_encoded['Risk']  # ou 'Risk_good'
df_encoded_sem_risk = df_encoded.drop(columns=['Risk'])
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df_encoded_sem_risk), columns=df_encoded_sem_risk.columns)
df_imputed['Risk'] = target_col.reset_index(drop=True)
for col in colunas_cat:
    le = label_encoders[col]
    df_imputed[col] = df_imputed[col].round(0).astype(int)
    df_imputed[col] = le.inverse_transform(df_imputed[col])
df = df_imputed

#GET_DUMMIES - TRANSFORMA OS DADOS EM BINARIO (UTIL A DT)'''
df = pd.get_dummies(df, drop_first=True)

#SEPARA O DADO ALVO DOS DEMAIS, EM SEGUIDA SPLIT NA PROPORÇÃO 70/30 TREINAMENTO/TESTE E DT
X = df.drop('Risk_good', axis=1)  
y = df['Risk_good']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

''' GERA A DT
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=X.columns, class_names=['Bad', 'Good'], filled=True)
plt.show()'''

# Calcula a importância por coluna
importances = pd.Series(dt.feature_importances_, index=X.columns)

# Define quais são as variáveis originais categóricas
variaveis_categoricas = ['Sex', 'Job', 'Purpose', 'Housing', 'Saving accounts', 'Checking account']

# Agrupar importâncias
importancia_agrupada = {}

# Agrupa todas as colunas dummies relacionadas à variável categórica
for var in variaveis_categoricas:
    colunas_relacionadas = [col for col in X.columns if col.startswith(var)]
    importancia_agrupada[var] = importances[colunas_relacionadas].sum()

# Adiciona as variáveis numéricas diretamente
for col in X.columns:
    if not any(col.startswith(cat) for cat in variaveis_categoricas):
        importancia_agrupada[col] = importances[col]

# Converte para Series e plota
importancia_agrupada = pd.Series(importancia_agrupada).sort_values(ascending=True)
'''
# Gráfico
plt.figure(figsize=(10, 6))
importancia_agrupada.plot(kind='barh')
plt.title("Importância dos Atributos (Agrupada)")
plt.xlabel("Importância")
plt.tight_layout()
plt.show()
'''
# RESULTADOS, ACURACIA, MATRIZ DE CONFUSÃO E RELATÓRIO
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
