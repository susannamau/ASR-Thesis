# Obiettivo: dataframe con parola, vettore, 0/1 per calcolo silhouette score

# Importazione modello come keyed vectors
import gensim
from gensim.models import KeyedVectors, Word2Vec
import pandas as pd
from gensim.test.utils import datapath
import sklearn
from sklearn import metrics
import numpy

model = KeyedVectors.load_word2vec_format("/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_100_5_0.05_esco.vec", encoding='utf8')
#model = fasttext.load_model("/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_100_5_0.01_esco.bin")
#print(model['computer'])

#vocab=gensim.models.FastText.build_vocab('/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito_20righe.csv')

# Importazione lista skills e corpus e creazione liste di parole
file_skills=pd.read_csv("/home/maugeri/ojvs-skills/Paper/my_exp/lista_skills_finale.csv", encoding='utf-8', sep="\n", header=0)
file_corpus=pd.read_csv("/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv", encoding='utf-8', sep="\n", header=0)

ski=file_skills["skills"]
cor=file_corpus["corpus"]
#print(cor[:20])
cor=cor[:1000]

lista_skills=[]
for skill in ski:
    if skill not in lista_skills:
        lista_skills.append(skill)
#print(lista_skills)

lista_no_skills=[]
for annuncio in cor:
    annuncio_spez=annuncio.split(" ")
    for parola in annuncio_spez:
        if parola not in lista_skills:
            if parola not in lista_no_skills:
                lista_no_skills.append(parola)
#print(lista_no_skills)

# ------------------------------------------------------------------

# Creazione dataframe per calcolo SS
#word_vectors= model.wv #coppie parola-vettore
words=[]
for w in lista_skills:
    if w in model:
        vettore=model[w]
        col_vettore=[]
        for dimensione in vettore:
            col_vettore.append(dimensione)
        col_vettore=tuple(col_vettore)
        oggetto=(w, col_vettore, "1") #tupla
        words.append(oggetto)
        #print(w)
#print(words)
#print(len(lista_skills))
#print(lista_skills)
#non si può appendere direttamente la tupla perché da errore, per questo creo oggetto
#se metto word_vectors[w] dentro la tupla, esce anche dtype=float32, mentre se lo richiedo separato no
#vettore esce giusto senza dtype=float32
#oggetto ha anche dtype=float32

for w in lista_no_skills:
    if w in model:
        vettore=model[w]
        col_vettore=[]
        for dimensione in vettore:
            col_vettore.append(dimensione)
        col_vettore=tuple(col_vettore)
        oggetto=(w, col_vettore, "0")
        words.append(oggetto)
        #print(vettore) #così non appare
        #print(oggetto) #così appare
#print(words[1])
#print(words[-1])
# words è una lista di tuple che si può trasformare in dataframe con dataframe=pd.DataFrame(words)

dataframe_words=pd.DataFrame(words)

dataframe_words2 = pd.DataFrame(dataframe_words[1].tolist())
#print(dataframe_words2) # solo vettori

frames = [dataframe_words[0], dataframe_words2, dataframe_words[2]]
result = pd.concat(frames, axis=1)

indice=[]
for i in range(0,102,1):
    indice.append(i)
#print(indice)

result.columns=indice
#print(result)

# --------------------------------------------------------

# CAMPIONAMENTO DI TOT SKILLS E NON SKILLS
# Devo campionare lo stesso numero di skills e non skills

n=600
sample_skills=result[result[101]=='1'].sample(n)
#print(sample_skills)
sample_non_skills=result[result[101]=='0'].sample(n)
#print(sample_non_skills)

sample_n=[sample_skills, sample_non_skills]
sample_da_usare = pd.concat(sample_n)
#print(sample_da_usare)
### SI!!!!!!!

#---------------------------------------------------------

# CALCOLO DEL SILHOUETTE SCORE

x = sample_da_usare.loc[:, 1:100]
#print(x)

score= sklearn.metrics.silhouette_score(x, metric='cosine', labels=sample_da_usare.loc[:, 101])
print(score)

# ---------------------------------------------------