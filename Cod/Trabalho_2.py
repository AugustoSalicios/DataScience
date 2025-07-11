from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind
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
#CONVERTER PARA NÚMEROS
X_train = X_train.astype(int)
X_test = X_test.astype(int)
X_train_final = X_train.copy()

#DT
dt_podada = DecisionTreeClassifier(
    max_depth=5,              
    min_samples_leaf=15,
    class_weight='balanced',
    random_state=42
)
dt_podada.fit(X_train, Y_train) 
Y_pred = dt_podada.predict(X_test)

#Avaliação de variavel DT
scores_dt = pd.Series(dt_podada.feature_importances_, index=X_train.columns, name='DecisionTree_Importance')
# 2. FAÇA A JUNÇÃO (AGRUPAMENTO) AGORA, DEPOIS DE OBTER OS SCORES
print("Agregando scores das features...")
# Identifica os nomes originais das features (ex: 'Purpose_car' -> 'Purpose')
nomes_originais = [col.split('_')[0] if '_' in col else col for col in X_train.columns]
# Agrupa os scores pelo nome original e soma
scores_dt_agregados = scores_dt.groupby(nomes_originais).sum()
scores_dt_agregados.name = 'DecisionTree_Importance'
# 3. Exiba o resultado final agregado
print("\n--- Importância Agregada das Features (DT Podada) ---")
print(scores_dt_agregados.sort_values(ascending=False))

#Avaliação Teste de Estudent
# Lista para guardar os scores (t-statistic) de cada feature
t_test_scores = []
# Loop através de cada coluna (feature) em X_train_final
for feature in X_train_final.columns:    
    # Separa os dados da feature em dois grupos, baseados no Y_train
    group_good = X_train_final[Y_train == 'good'][feature]
    group_bad = X_train_final[Y_train == 'bad'][feature]    
    # Roda o teste t para amostras independentes
    # 'equal_var=False' realiza o teste de Welch, que é mais robusto
    # e não assume que os dois grupos têm a mesma variância.
    t_stat, p_value = ttest_ind(group_good, group_bad, equal_var=False)    
    # Guardamos o valor absoluto do t-statistic.
    # Quanto maior o valor, mais diferentes são as médias dos grupos.
    t_test_scores.append(abs(t_stat))
# Converte a lista de scores em uma Series do pandas para fácil manipulação
scores_ttest = pd.Series(t_test_scores, index=X_train_final.columns, name='T-Test_Score')
nomes_originais_ttest = [col.split('_')[0] if '_' in col else col for col in scores_ttest.index]
ranking_ttest_agrupado = scores_ttest.groupby(nomes_originais).max()
ranking_ttest_agrupado.name = 'T-Test_Aggregated_Max_Score'

print("\n--- Scores Agregados do Teste t (pelo Máximo) ---")
print(ranking_ttest_agrupado.sort_values(ascending=False))

#Avaliação usando RFE com Regressão Logistica
# 1. Crie a instância do modelo base (estimador)
# Usamos max_iter=1000 para garantir que o modelo convirja sem erros.
estimator = LogisticRegression(max_iter=1000, random_state=42)
# 2. Crie a instância do RFE
# n_features_to_select=1 significa que queremos que ele rode até sobrar apenas 1 feature,
# o que nos dará um ranking completo de todas.
selector_rfe = RFE(estimator, n_features_to_select=1, step=1)
# 3. Treine o RFE com os dados
selector_rfe.fit(X_train_final, Y_train)
# 4. Extraia o ranking das features
# O atributo .ranking_ nos diz a ordem de eliminação.
# 1 = feature mais importante (a última a ser removida)
# 2 = a segunda mais importante, e assim por diante.
ranking_rfe_bruto = pd.Series(selector_rfe.ranking_, index=X_train_final.columns)

nomes_originais_rfe = [col.split('_')[0] if '_' in col else col for col in ranking_rfe_bruto.index]

# Agrupando os resultados gerados pelo RFE
ranking_rfe_agregado = ranking_rfe_bruto.groupby(nomes_originais_rfe).min()
ranking_rfe_agregado.name = 'RFE_Aggregated_Ranking'
# Visualizar o resultado final
print("\n--- Ranking Agregado das Features pelo RFE (Menor = Mais Importante) ---")
print(ranking_rfe_agregado.sort_values(ascending=True))
'''
print('Acuracia: ', accuracy_score(Y_test, Y_pred))
print('\nMatriz Confus: ', )
print(confusion_matrix(Y_test, Y_pred))
print('\nRelatorio de Classificação: ')
print(classification_report(Y_test, Y_pred))
'''