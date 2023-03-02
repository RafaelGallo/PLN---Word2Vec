#!/usr/bin/env python
# coding: utf-8

# # Treinamento word2vec
# 
# **Base dados**
# 
# - http://www.nilc.icmc.usp.br/embeddings

# In[1]:


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
import spacy
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


# In[2]:


# Baixando modelo pré-treino
get_ipython().system('python -m spacy download pt_core_news_sm')


# In[3]:


# Importando modelo pré-treinado
nlp = spacy.load("pt_core_news_sm")
nlp


# # Base dados

# In[4]:


data1 = pd.read_csv("data/treino.csv")


# In[5]:


data2 = pd.read_csv("data/teste.csv")


# In[6]:


# Visualizando 10 primeiros dados
data1.head(10)


# In[7]:


# Visualizando 10 últimos dados
data1.tail()


# In[8]:


# Linhas colunas
data1.shape


# In[9]:


# Info dados
data1.info()


# In[10]:


# Tipo dados
data1.dtypes


# In[11]:


# Amostra dados
data1.text.sample(150)


# In[12]:


texto = "Rio de Janeiro é uma cidade maravilhosa"
doc = nlp(texto)
doc


# In[13]:


textos_para_tratamento = (titulos.lower() for titulos in data1["title"])
textos_para_tratamento


# In[14]:


def trata_textos(doc):
    tokens_validos = []
    for token in doc:
        e_valido = not token.is_stop and token.is_alpha
        if e_valido:
            tokens_validos.append(token.text)

    if len(tokens_validos) > 2:
        return  " ".join(tokens_validos)

texto = "Rio de Janeiro 1231231 ***** @#$ é uma cidade maravilhosa!"
doc = nlp(texto)
trata_textos(doc)


# In[15]:


texto = "Rio de Janeiro 1231231 ***** @#$ é uma cidade maravilhosa!"
doc = nlp(texto)
trata_textos(doc)


# In[16]:


from time import time

t0 = time()
textos_tratados = [trata_textos(doc) for doc in nlp.pipe(textos_para_tratamento,
                                                        batch_size = 1000,
                                                        n_process = -1)]

tf = time() - t0

titulos_tratados = pd.DataFrame({"titulo": textos_tratados})
titulos_tratados.head()


# # Modelo word2vec

# In[17]:


#size = 300

from gensim.models import Word2Vec

model_word2vec = Word2Vec(sg = 0,
                          window = 2,
                          min_count = 5,
                          alpha = 0.03,
                          min_alpha = 0.007)


# In[18]:


print(len(titulos_tratados))


# In[19]:


titulos_tratados = titulos_tratados.dropna().drop_duplicates()
print(len(titulos_tratados))


# In[20]:


lista_lista_tokens = [titulo.split(" ") for titulo in titulos_tratados.titulo]
lista_lista_tokens


# # Treinamento modelo word2vec

# In[21]:


import logging

# Treinamento das palavras
logging.basicConfig(format="%(asctime)s : - %(message)s", level = logging.INFO)

# Modelo 
model_word2vec2 = Word2Vec(sg = 0,
                           window = 2,
                           min_count = 5,
                           alpha = 0.03,
                           min_alpha = 0.007)

# Visualizando modelo
model_word2vec2.build_vocab(lista_lista_tokens, progress_per=5000)


# In[22]:


# Modelo 02 - treinamento word2vec

model_word2vec2.train(lista_lista_tokens, 
                 total_examples=model_word2vec2.corpus_count,
                 epochs = 30)


# In[23]:


# Visualizando os textos similar 1
model_word2vec2.wv.most_similar("google")


# In[24]:


# Visualizando os textos similar 2
model_word2vec2.wv.most_similar("microsoft")


# In[25]:


# Visualizando os textos similar 3
model_word2vec2.wv.most_similar("barcelona")


# In[26]:


# # Visualizando os textos similar 4
model_word2vec2.wv.most_similar("messi")


# In[27]:


# Visualizando os textos similar 5
model_word2vec2.wv.most_similar("gm")


# # Treinamento word2vec -  Skip-Gram

# In[31]:


#Treinamento do modelo Skip-Gram
w2v_modelo_sg = Word2Vec(sg = 1,
                      window = 5,
                      min_count = 5,
                      alpha = 0.03,
                      min_alpha = 0.007)

# Vocabularios modelos
w2v_modelo_sg.build_vocab(lista_lista_tokens, progress_per=5000)

# Treinamento modelo
w2v_modelo_sg.train(lista_lista_tokens, 
                 total_examples=w2v_modelo_sg.corpus_count,
                 epochs = 100)


# In[32]:


# Visualizando os textos similar
w2v_modelo_sg.wv.most_similar("google")


# In[39]:


# Visualizando os textos similar
w2v_modelo_sg.wv.most_similar("microsoft")


# In[43]:


# Visualizando os textos similar
w2v_modelo_sg.wv.most_similar("coca")


# In[33]:


# Visualizando os textos similar
w2v_modelo_sg.wv.most_similar("gm")


# In[35]:


# Visualizando os textos similar
w2v_modelo_sg.wv.most_similar("bmw")


# # Salvando modelos

# In[37]:


model_word2vec.wv.save_word2vec_format("data/modelo_cbow.txt", binary=False)


# In[38]:


w2v_modelo_sg.wv.save_word2vec_format("data/modelo_skipgram.txt", binary=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




