#!/usr/bin/env python
# coding: utf-8

# # Introdução word2vec
# 
# **Base dados**
# 
# - http://www.nilc.icmc.usp.br/embeddings

# In[148]:


import unicodedata
import sys
import string
import time
import csv
import nltk
import re

import numpy as np
import pandas as pd
import seaborn as sns
import warnings

import nltk
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
nltk.download("stopwords")
nltk.download('punkt')

from pylab import rcParams
import matplotlib.pyplot as plt
import matplotlib as m
import matplotlib as mpl

warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')
m.rcParams['axes.labelsize'] = 14
m.rcParams['xtick.labelsize'] = 12
m.rcParams['ytick.labelsize'] = 12
m.rcParams['text.color'] = 'k'
rcParams['figure.figsize'] = 18, 8

get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Rafael Gallo" --iversions')

from platform import python_version
print('Versão Jupyter Notebook neste projeto:', python_version())


# # Base dados

# In[2]:


data1 = pd.read_csv("data/treino.csv")


# In[3]:


data2 = pd.read_csv("data/teste.csv")


# In[4]:


# Visualizando 10 primeiros dados
data1.head(10)


# In[5]:


# Visualizando 10 últimos dados
data1.tail()


# In[6]:


# Linhas colunas
data1.shape


# In[7]:


# Info dados
data1.info()


# In[8]:


# Tipo dados
data1.dtypes


# In[13]:


# Amostra dados
data1.text.sample(25)


# In[15]:


# Vitorização dados
from sklearn.feature_extraction.text import CountVectorizer

texto =[
        "tenha um bom dia",
        "tenha um péssimo dia",
        "tenha um ótimo dia",
        "tenha um dia ruim python java alura caelum papa"
    
]

# Criando uma instância
vet = CountVectorizer()

# Treinamento modelo
vet_fit = vet.fit(texto)


# In[16]:


# Visualizando os vocabularios
vet.vocabulary_


# In[17]:


# Transformando vetores para sentimento
vet_bom = vet.transform(["bom"])
print(vet_bom.toarray())


# # Base dados - modelo nlp

# In[18]:


with open("models/cbow_s300.txt") as x:
    for linha in range(10):
        print(next(x))


# # Modelo Word2vec

# In[20]:


get_ipython().run_cell_magic('time', '', '\n# Importando modelo Word2Vec\nfrom gensim.models import KeyedVectors\n\n# Criando uma instância modelo\nmodel = KeyedVectors.load_word2vec_format("models/cbow_s300.txt")')


# In[21]:


get_ipython().run_cell_magic('time', '', '\n# Visualizando textos similar 1\nmodel.most_similar("china")')


# In[22]:


get_ipython().run_cell_magic('time', '', '\n# Visualizando textos similar 2\nmodel.most_similar("itália")')


# In[23]:


get_ipython().run_cell_magic('time', '', '\n# Visualizando textos similar 3\nmodel.most_similar("china")')


# In[29]:


get_ipython().run_cell_magic('time', '', '\n# Visualizando textos similar 4\nmodel.most_similar("russia")')


# In[30]:


get_ipython().run_cell_magic('time', '', '\n# Visualizando textos similar 5\nmodel.most_similar("alemanha")')


# In[24]:


get_ipython().run_cell_magic('time', '', '\n# Textos positivos\nmodel.most_similar(positive=["brasil", "argentina"])')


# In[25]:


get_ipython().run_cell_magic('time', '', '\n#nuvens -> nuvem : estrelas -> estrela\n#nuvens + estrela - nuvem = estrelas\n\nmodel.most_similar(positive=["nuvens", "estrela"], \n                   negative=["nuvem"])')


# In[26]:


get_ipython().run_cell_magic('time', '', '\n# Visualizando textos positivos negativos\nmodel.most_similar(positive=["professor", "mulher"], \n                    negative=["homem"])')


# # Vetorização textos em geral

# In[27]:


data1.title.loc[100]


# In[63]:


# Tokenização textos

def token(text):
    text = text.lower()
    data_list = []
    
    for token in nltk.word_tokenize(text):
        if token in string.punctuation: continue
        data_list.append(token)
        
    return data_list

# Função vetorização textos
def combinacao_de_vetores_por_soma(palavras_numeros):
    vetor_resultante = np.zeros((1,300))
    for pn in palavras_numeros:
        try:
            vetor_resultante += model.get_vector(pn)

        except KeyError:
            if pn.isnumeric():
                pn = "0"*len(pn)
                vetor_resultante += model.get_vector(pn)
                
            else:
                vetor_resultante += model.get_vector("unknown")
    
    return vetor_resultante

# Matriz dos vetores
def matriz_vetores(textos):
    x = len(textos)
    y = 300
    matriz = np.zeros((x,y))

    for i in range(x):
        palavras_numeros = token(textos.iloc[i])
        matriz[i] = combinacao_de_vetores_por_soma(palavras_numeros)

    return matriz


# In[51]:


# Visualizando token
token("Texto Exemplo, 1234.")


# In[52]:


print(vetor_texto) = token("texto exemplo caelumx")
pn


# In[59]:


palavras_numeros = token("texto exemplo caelumx")


# In[68]:


vetor_texto = combinacao_de_vetores_por_soma(palavras_numeros)


# In[69]:


matriz_vetores_treino = matriz_vetores(data1.title)
matriz_vetores_treino


# In[70]:


matriz_vetores_teste = matriz_vetores(data2.title)
matriz_vetores_teste


# # Modelo machine learning

# In[74]:


get_ipython().run_cell_magic('time', '', '\n# Importando biblioteca\nfrom sklearn.linear_model import LogisticRegression\n\n# Criando modelo modelo regressão logistica\nmodel_logistic = LogisticRegression(max_iter = 500)\n\n# Criando treinamento modelo\nmodel_logistic_fit = model_logistic.fit(matriz_vetores_treino, data1.category)')


# In[76]:


# Score modelo
model_logistic_score = model_logistic.score(matriz_vetores_treino, data1.category)
model_logistic_score


# In[77]:


# Previsão modelo
model_logistic_pred = model_logistic.predict(matriz_vetores_teste)
model_logistic_pred


# In[130]:


from sklearn.metrics import accuracy_score

acuracia_model_logistic = accuracy_score(data2.category, model_logistic_pred)

print("Accuracy - Logistic Regression: %.2f" % (acuracia_decision_tree * 100))


# In[132]:


from sklearn.metrics import confusion_matrix

matrix_confusion_1 = confusion_matrix(data2.category, model_logistic_pred)
matrix_confusion_1


# In[144]:


ax= plt.subplot()
sns.heatmap(matrix_confusion_1, annot=True, ax = ax, fmt = ".1f", cmap = 'plasma'); 

ax.set_xlabel('');
ax.set_ylabel(''); 
ax.set_title('Confusion Matrix - Logistic Regression'); 


# In[81]:


# Avaliação modelo
from sklearn.metrics import classification_report

# Criando cristancia avaliador
class_report = classification_report(data2.category, model_logistic_pred)
print(class_report)


# # Modelo 02 - Decision Tree

# In[96]:


get_ipython().run_cell_magic('time', '', '\n# Importando biblioteca\nfrom sklearn.tree import DecisionTreeClassifier \n\n# Criando modelo modelo Decision Tree\nmodel_dtc = DecisionTreeClassifier(max_depth = 4, \n                                   random_state = 0)\n\n# Criando treinamento modelo\nmodel_dtc_fit = model_dtc.fit(matriz_vetores_treino, data1.category)')


# In[97]:


# Score modelo
model_dtc_score = model_dtc.score(matriz_vetores_treino, data1.category)
print("Modelo - Decision Tree Classifier: %.2f" % (model_dtc_score * 100))


# In[98]:


# Previsão modelo
modelo_arvore_cla_1_predict = model_dtc.predict(matriz_vetores_teste)
modelo_arvore_cla_1_predict


# In[99]:


# Probabilidade
modelo_arvore_cla_1_prob = model_dtc.predict_proba(matriz_vetores_teste)
modelo_arvore_cla_1_prob


# In[104]:


from sklearn.metrics import accuracy_score

acuracia_decision_tree = accuracy_score(data2.category, modelo_arvore_cla_1_predict)

print("Accuracy - Decision Tree: %.2f" % (acuracia_decision_tree * 100))


# In[113]:


class_report = classification_report(data2.category, modelo_arvore_cla_1_predict)
print("Modelo - Decision Tree")
print("\n")
print(class_report)


# In[105]:


from sklearn.metrics import confusion_matrix

matrix_confusion_1 = confusion_matrix(data2.category, modelo_arvore_cla_1_predict)


# In[145]:


ax= plt.subplot()
sns.heatmap(matrix_confusion_1, annot=True, ax = ax, fmt = ".1f", cmap = 'plasma'); 

ax.set_xlabel('');
ax.set_ylabel(''); 
ax.set_title('Confusion Matrix - Decision Tree Classifier'); 
#ax.xaxis.set_ticklabels(["Positivo", "Negativo"]); ax.yaxis.set_ticklabels(["Positivo", 'Negativo']);


# # Modelo 03 - KNN

# In[111]:


get_ipython().run_cell_magic('time', '', '\n# Importando biblioteca\nfrom sklearn.neighbors import KNeighborsClassifier\n\n# Criando uma instância\nmodel_knn = KNeighborsClassifier()\n\n# Treinamento modelo\nmodel_knn_fit = model_knn.fit(matriz_vetores_treino, data1.category)')


# In[112]:


# Score modelo
model_knn_score = model_knn.score(matriz_vetores_treino, data1.category)
print("Modelo - K-NN: %.2f" % (model_knn_score * 100))


# In[114]:


# Previsão do modelo do k-nn

model_knn_pred = model_knn.predict(matriz_vetores_teste)
model_knn_pred


# In[115]:


accuracy_knn = accuracy_score(data2.category, model_knn_pred)
print("Acurácia - K-NN: %.2f" % (accuracy_knn * 100))


# In[146]:


matrix_confusion_3 = confusion_matrix(data2.category, model_knn_pred)

ax = plt.subplot()
sns.heatmap(matrix_confusion_3, annot=True, ax = ax, fmt = ".1f", cmap = 'plasma'); 

ax.set_xlabel('');
ax.set_ylabel(''); 
ax.set_title('Confusion Matrix - K-NN'); 


# In[117]:


classification = classification_report(data2.category, model_knn_pred)
print("Modelo 04 - K-NN")
print()
print(classification)


# # Modelo 04 - Random forest

# In[118]:


get_ipython().run_cell_magic('time', '', '# Modelo 04 - Random forest\n\n# Importando biblioteca\nfrom sklearn.ensemble import RandomForestClassifier\n\n# max_depth - determinando total de árvore, random_state 0\nmodel_random_forest = RandomForestClassifier(max_depth = 2, random_state = 0) \n\n# Dados de treino, teste de x, y\nmodel_random_forest_fit = model_random_forest.fit(matriz_vetores_treino, data1.category)')


# In[119]:


# Valor da Accuracy do algoritmo
model_random_forest_score = model_random_forest.score(matriz_vetores_treino, data1.category) 
print("Score - Modelo random forest: %.2f" % (model_random_forest_score * 100))


# In[120]:


# Previsão do modelo
model_random_forest_regressor_pred = model_random_forest.predict(matriz_vetores_teste)
model_random_forest_regressor_pred


# In[121]:


# Accuracy model
accuracy_random_forest = accuracy_score(data2.category, model_random_forest_regressor_pred)
print("Accuracy - Random forest: %.2f" % (accuracy_random_forest * 100))


# In[147]:


matrix_confusion_4 = confusion_matrix(data2.category, model_random_forest_regressor_pred)

ax = plt.subplot()
sns.heatmap(matrix_confusion_4, annot=True, ax = ax, fmt = ".1f", cmap = 'plasma'); 

ax.set_xlabel('');
ax.set_ylabel(''); 
ax.set_title('Confusion Matrix - Random forest'); 


# In[123]:


classification = classification_report(data2.category, model_random_forest_regressor_pred)
print("Modelo 04 - Random forest")
print()
print(classification)


# # Resultado final

# In[136]:


# Resultados - Modelos machine learning

modelos = pd.DataFrame({
    
    "Models" :["Regressão logistica", 
                "K-NN", 
                "Random Forest", 
                "Decision Tree"],

    "Acurácia" :[acuracia_decision_tree,
                 acuracia_model_logistic, 
                 accuracy_knn,
                 accuracy_random_forest]})

modelos.sort_values(by = "Acurácia", ascending = False)


# In[143]:


## Salvando modelo M.L word2vec

import pickle
 
with open('modelo_arvore_cla_1_predict.pkl', 'wb') as file:
    pickle.dump(modelo_arvore_cla_1_predict, file)

with open('model_knn_pred.pkl', 'wb') as file:
    pickle.dump(model_knn_pred, file)

with open('model_random_forest_regressor_pred.pkl', 'wb') as file:
    pickle.dump(model_random_forest_regressor_pred, file)


# In[ ]:




