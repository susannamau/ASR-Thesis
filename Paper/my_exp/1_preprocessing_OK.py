import numpy as np
import spacy
import re
#import os
import pandas as pd
#import itertools
from tqdm import tqdm_notebook
#import matplotlib.pyplot as plt
import warnings
# import umap.umap_ as umap # pip install umap-learn
#import seaborn as sns

import gensim.parsing as gm
from gensim.parsing.preprocessing import preprocess_string
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

import nltk
from nltk.stem import WordNetLemmatizer
#nltk.download('stopwords')
from nltk.corpus import stopwords
#nltk.download('wordnet')

from tqdm import tqdm
import html

#------------------------------------------------------------------------------------------

### IMPORTAZIONE DEI DATI
file=pd.read_csv("/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train.csv", encoding='utf-8', engine='python', error_bad_lines=False, header=None)
#print(type(file)) #come atteso è un dataframe OK
#print(file[0]) #è la colonna della descrizione

description = file[0]
#description=description[:10]
#print(description)
#print(description.head(5))

#-------------------------------------------------------------------------------------------

### PREPROCESSING
def text_preproc(text: str):
    my_filter = [
        lambda x: x.lower(), # rende tutto minuscolo
        gm.strip_punctuation, # rimuove punteggiatura
        gm.strip_multiple_whitespaces, # rimuove spaziatura
        gm.strip_numeric, # rimuove numeri
        gm.remove_stopwords, # rimuove stopwords
        gm.strip_short,
        gm.strip_tags,
        #gm.stem_text # riporta le parole dalla forma flessa alla loro radice 
        ##fare funzione lemmatizzazione
    ]
    return preprocess_string(text, filters=my_filter)

# Funzione per rimozione caratteri html
def remove_html_tags(text):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# Funzione per lemmatizzazione
lemmatizer = WordNetLemmatizer()

# Rimuovo html e rendo tutto minuscolo, funzioni su tutta la stringa dell'annuncio
lista=[] #lista in cui ogni riga è un elemento e di tipo stringa
for line in description:
    riga=str(line)
    riga=riga.lower()   #tutti i caratteri sono resi minuscoli
    riga=remove_html_tags(riga)
    lista.append(riga)
#print(lista[:3]) OK
# lista: ogni elemento è una stringa che contiene una riga del dataset

lista = [re.sub(r'[^a-z\s\+\#]', ' ', x) for x in lista]
#print(lista[:3])

# Applico la funzione text_preproc e lemmatizzo ogni termine, funzioni sulle singole parole degli annunci
lista2=[] #lista di liste degli annunci spezzettati per parole (lemmatizzate)
for stringa in lista:
    riga_preproc=text_preproc(stringa)
    lista_lemm=[]
    for parola in riga_preproc:
        parola_lemm=lemmatizer.lemmatize(parola)
        lista_lemm.append(parola_lemm)
        #print(parola_lemm)
    lista2.append(lista_lemm)
    #print(lista2)
#lista2: ogni elemento è una lista che contiene le singole parole 'significative' di un annuncio
# il risultato è una lista di liste: ogni lista interna rappresenta un annuncio di lavoro
# e i suoi elementi solo le singole parole che hanno significato, no stopwords ecc (lemmatizzate)
#print(lista2[:5])

#corpus_in_lista è una lista di stringhe, dove ogni stringa è un annuncio di lavoro con le parole lemmatizzate ecc
corpus_in_lista=[]
for lista in lista2:
    stringa_annuncio=''
    for parola in lista:
        stringa_annuncio=stringa_annuncio+' '+parola
    corpus_in_lista.append(stringa_annuncio)
#print(corpus_in_lista)

# Prova lemmatizzazione
#print("rocks :", lemmatizer.lemmatize("rocks")) 
#print("corpora :", lemmatizer.lemmatize("corpora")) 

# Rimozione di ulteriori stopwords relative al lessico degli annunci di lavoro
stop_words_ojv=['job', 'salary', 'europe', 'junior', 'senior', 'opportunity', 'company', 'contact',
'skills', 'per annum', 'job vacancy', 'click', 'site', 'com', 'work', 'apply', 'note', 
'payment', 'recruiter', 'jobseeker','experience', 'client', 'money', 'team', 'bonus', 'employee',
'employer', 'recruitment', 'job training', 'age',  'spain', 'italy', 'uk', 'germany', 'france', 'portugal', 'employment', 
'scotland', 'london', 'seeking', 'experienced', 'country', 'city', 'centre', 'religion',
'gender', 'orientation', 'person', 'marital status', 'color', 'disability', 'telephone', 'email', 'career',
'jobseekers', 'required', 'england', 'need', 'role', 'tell', 'let', 'position', 'seeking', 'www', 'join', 'vacancy', 'year']
for annuncio in lista2:
    for parola in annuncio:
        if parola in stop_words_ojv:
            annuncio.remove(parola)
# lista2=[annuncio.remove(parola) for annuncio in lista2 for parola in annuncio if parola in stop_words_ojv] #non funziona, restituisce none
print(lista2[:5])

#-------------------------------------------------------------------------------------------

# UNIFICAZIONE DEI 5 FILE CON DIVERSE SKILLS E SOSTITUZIONE DEL CORPUS
# TROVARE E SOSTITUIRE SKILLS DA UNA LISTA DATA
# Prendo la lista di skills, metto insieme quelle che sono formate da parole multiple con un _,
# le cerco nel corpus e le sostituisco.
file_skills0=pd.read_csv("/home/maugeri/ojvs-skills/Paper/my_exp/skills_txt/mapping_id_skills.csv", sep=',', encoding='utf-8')
file_skills1=pd.read_csv("/home/maugeri/ojvs-skills/Paper/my_exp/skills_txt/skills_ict.csv", sep=',', encoding='utf-8')
file_skills2=pd.read_csv("/home/maugeri/ojvs-skills/Paper/my_exp/skills_txt/skills_no_ict.csv", sep=',', encoding='utf-8')
file_skills3=pd.read_csv("/home/maugeri/ojvs-skills/Paper/my_exp/skills_txt/skills_stack.csv", sep=',', encoding='utf-8')
file_skills4=open("/home/maugeri/ojvs-skills/Paper/my_exp/skills_txt/skills_tot_esco.txt", 'r')

skills0=file_skills0['escoskill_level_3']
#print(skills.head(10))
lista_skills0=[]
for line in skills0:
    riga=str(line)
    nuova_riga=re.sub(r"\s+", "_", riga)
    #print(nuova_riga)
    lista_skills0.append(nuova_riga)
#print(lista_skills0)

skills1=file_skills1['0']
#print(skills1)
lista_skills1=[]
for line in skills1:
    riga=str(line)
    nuova_riga=re.sub(r"\s+", "_", riga)
    #print(nuova_riga)
    lista_skills1.append(nuova_riga)
#print(lista_skills1) #lista di stringhe, ogni stringa è una skill nel formato skill_skill

skills2=file_skills2['0']
#print(skills2)
lista_skills2=[]
for line in skills2:
    riga=str(line)
    nuova_riga=re.sub(r"\s+", "_", riga)
    #print(nuova_riga)
    lista_skills2.append(nuova_riga)
#print(lista_skills2) #lista di stringhe, ogni stringa è una skill nel formato skill_skill

skills3=file_skills3['skill']
#print(skills3)
lista_skills3=[]
for line in skills3:
    riga=str(line)
    nuova_riga=re.sub(r"\s+", "_", riga)
    #print(nuova_riga)
    lista_skills3.append(nuova_riga)
#print(lista_skills3) #lista di stringhe, ogni stringa è una skill nel formato skill_skill

# Le tre liste lista_skills1, lista_skills2 e lista_skills3 sono omogenee.

testo=file_skills4.read()
lista_skills4=testo.split('\n')
#print(lista_skills4)

# Rimozione stringhe vuote:
for stringa in lista_skills0:
    if stringa=='':
        lista_skills0.remove(stringa)
for stringa in lista_skills1:
    if stringa=='':
        lista_skills1.remove(stringa)
for stringa in lista_skills2:
    if stringa=='':
        lista_skills2.remove(stringa)
for stringa in lista_skills3:
    if stringa=='':
        lista_skills3.remove(stringa)
for stringa in lista_skills4:
    if stringa=='':
        lista_skills4.remove(stringa)
#print(lista_skills4) #qui ero sicura che l'ultima stringa fosse vuota

# Ora le 4 liste sono omogenee.
# Aggiungo anche la lista che avevo già considerato nel file forse2.py.

# Unisco le skills in modo tale che appaiano una sola volta in una lista che le contiene tutte
lista_skills_finale=[]
for skill in lista_skills0:
    if skill not in lista_skills_finale:
        lista_skills_finale.append(skill)
for skill in lista_skills1:
    if skill not in lista_skills_finale:
        lista_skills_finale.append(skill)
for skill in lista_skills2:
    if skill not in lista_skills_finale:
        lista_skills_finale.append(skill)
for skill in lista_skills3:
    if skill not in lista_skills_finale:
        lista_skills_finale.append(skill)
for skill in lista_skills4:
    if skill not in lista_skills_finale:
        lista_skills_finale.append(skill)
#print(lista_skills_finale)

contatore=[]
for skill in lista_skills_finale:
    parole_skill=skill.split('_') #è una lista
    skill_stringa=''
    for parola in parole_skill:
        skill_stringa=skill_stringa+' '+parola #è una stringa unica
    for annuncio in corpus_in_lista:
        if skill_stringa in annuncio:
            annuncio.replace(skill_stringa, skill)
            if skill_stringa not in contatore:
                contatore.append(skill_stringa)
            #print(skill)        
#print(contatore)
#print(len(contatore))
#263 skills su 1411 sono state trovate nel corpus e sostituite FORSE NUMERI SBAGLIATI

#print(corpus_in_lista)

#-----------------------------------------------------------------------------------------------

# TROVARE E SOSTITUIRE OCCUPAZIONI DA UNA LISTA DATA

file=pd.read_csv("/home/maugeri/ojvs-skills/Paper/my_exp/occ_en.csv", sep=',', encoding='utf-8')
occupations=file['occupations']
#print(occupations)

lista_occupazioni_spazi=[]
for line in occupations:
    line=str(line)
    lista_occupazioni_spazi.append(line)
#print(lista_occupazione_spazi)

contatore=[]
for occupazione_spazio in lista_occupazioni_spazi:
    occupazione_underscore=re.sub(r"\s+", "_", line)
    for annuncio in corpus_in_lista:
        if occupazione_spazio in annuncio:
            annuncio.replace(occupazione_spazio, occupazione_underscore)
            if occupazione_spazio not in contatore:
                contatore.append(occupazione_spazio)
#print(contatore)
#print(len(contatore))
# 4025 occupazioni sono state trovate nel corpus e sostituite

# Spezzetto gli annunci in singole parole in modo da poter applicare phrases.
lista3=[]
for annuncio in corpus_in_lista:
    lista_annuncio=annuncio.split(' ')
    lista3.append(lista_annuncio)
#print(lista3)
#lista 3 è una lista di liste di parole con skills e occupazioni sostituite

for lista in lista3:
    for elemento in lista:
        if elemento == '':
            lista.remove(elemento)
print(lista3)

#--------------------------------------------------------------------------------------------

# CREAZIONE DEI BIGRAMMI
import spacy
#spacy.load('en')
from spacy.lang.en import English

bigram= Phrases(lista3, min_count=10, threshold=0.2) #questo è il training!

# Per stampare i brigrammi creati:
# for sent in lista2:
#     bigrams_ = [b for b in bigram[sent] if '_' in b ]
#     if bigrams_ != []:
#         print(bigrams_)

# Per sostituire i bigrammi nel corpus:
nuovo_corpus=[]
for annuncio in lista2:
    nuovo_annuncio=bigram[annuncio] #bigram prende in imput una lista di stringhe di singole parole
    #print(nuovo_annuncio)
    nuovo_corpus.append(nuovo_annuncio)
#print(nuovo_corpus)

# nuovo_corpus è una lista di liste. Ogni lista interna contiene le parole di un annuncio,
# dove i bigrammi sono stati sostituiti alle coppie frequenti.
# Quindi i bigrammi creati sono stati sostituiti alle coppie.
# lista2 --> nuovo_corpus 

# Ora devo creare degli ngrammi con n>2.
# Riapplico Phrases a nuovo_corpus per fargli cercare altri ngrammi frequenti.
ngram= Phrases(nuovo_corpus, min_count=10, threshold=0.2) #questo è il training!


# Questo mi serve per stampare i bigrammi e gli ngrammi creati
# for sent in nuovo_corpus:
#     ngrams_ = [b for b in ngram[sent] if '_' in b ]
#     if ngrams_ != []:
#         print(ngrams_)

# Per sostituirli nel corpus:
nuovo_nuovo_corpus=[]
for annuncio in nuovo_corpus:
    nuovo_annuncio=ngram[annuncio]
    nuovo_nuovo_corpus.append(nuovo_annuncio)
#print(nuovo_nuovo_corpus)

# nuovo_nuovo_corpus è una lista di liste. Ogni lista interna contiene le parole di un annuncio,
# dove i bigrammi e gli ngrammi sono stati sostituiti alle parole vicine di frequente.
# Quindi i bigrammi e gli ngrammi creati sono stati sostituiti alle coppie.
# lista2 --> nuovo_corpus --> nuovo_nuovo_corpus

#------------------------------------------------------------------------------------------------

# RIMOZIONE ULTERIORI STOPWORDS

stop_words_ojv.extend(['new', 'global', 'including', 'agency', 'strong', 'got', 'offer', 'posted', 'manchester', 'london'])
for annuncio in nuovo_nuovo_corpus:
    for parola in annuncio:
        if parola in stop_words_ojv:
            annuncio.remove(parola)

corpus_fin=[]
for lista in nuovo_nuovo_corpus:
    stringa_annuncio=""
    for parola in lista:
        stringa_annuncio=stringa_annuncio+' '+parola
    stringa_annuncio=stringa_annuncio.strip()
    corpus_fin.append(stringa_annuncio)
print(corpus_fin)
# corpus_fin è una lista di stringhe dove ogni stringa è un annuncio di lavoro preprocessato e dove
# le parola frequentemente vicine sono state sostituite da bigrammi ed ngrammi.

#------------------------------------------------------------------------------------------------

# ESPORTAZIONE DEL CORPUS PREPROCESSATO PER LA CREAZIONE DEI MODELLI
file=open('/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv', 'w')
file.write("corpus\n")
for annuncio in corpus_fin:
    #print(annuncio)
    file.write(str(annuncio)+'\n')
file.close()

file20=open('/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito_20righe.csv', 'w')
file20.write("corpus\n")
for annuncio in corpus_fin[:20]:
    file20.write(str(annuncio)+'\n')
file20.close()