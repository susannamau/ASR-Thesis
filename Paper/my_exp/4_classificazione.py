# IMPORT -------------------------------------------------------------------------------------------------------------------
# Importazione librerie utili
import gensim
from gensim.models import KeyedVectors
import pandas as pd
import re
from gensim.test.utils import datapath
import numpy as np
import random
#import shorttext
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import collections
from collections import Counter
from collections import defaultdict

import sklearn
from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, accuracy_score

from subprocess import check_output

import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, LSTM, Bidirectional, GRU, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Import embedding model
embeddings = KeyedVectors.load_word2vec_format("/home/maugeri/ojvs-skills/Paper/my_exp/output/cbow_300_20_0.01_esco.vec", encoding='utf8')
file_corpus=pd.read_csv("/home/maugeri/ojvs-skills/Paper/my_exp/dataset/corpus_train_finito.csv", encoding='utf-8', sep="\n", header=0)
file_skills=pd.read_csv("/home/maugeri/ojvs-skills/Paper/my_exp/lista_skills_finale.csv", encoding='utf-8', sep="\n", header=0)

no_skills=['programmer', 'tester', 'cto', 'dba', 'cio', 'ciso', 'ict_technician', \
'cdo', 'sysadmin', 'videographer', 'webmaster', 'audio_visual_technician', 'camera_operator', \
'ethical_hacker', 'growth_hacker', 'cameraman', 'camerawoman', 'taxonomist', 'projectionist', \
'developer', 'engineer', 'product', 'system', 'customer', 'deliver', 'analyst', 'candidate', 'office', \
'specialist', 'sector', 'contract', 'stakeholder', 'architect', \
'procedure', 'provider', 'leader', 'budget', 'client', \
'commerce', 'storage', 'workflow', 'partnership', 'hospital', 'opportunity', 'technician', \
'medical', 'stage', 'scientist', 'assistant', 'bank', 'student', 'selection', 'player', 'senior', 'director', \
'law', 'industrial', 'file', 'market_leader', 'workplace', 'script', 'transport', 'roadmap', \
'permanent_employment', 'respect', 'driver', 'privacy', 'public_sector', 'private', \
'resident', 'machine', 'state_art', 'salary', 'workforce', 'internet', 'startup', 'conference', 'negotiation', \
'sport', 'room', 'supervisor', 'email', 'backup', 'membership', 'legislation', 'version_control', \
'business_relation_vacancy', 'training_course', 'pharmaceutical', 'product_owner', 'food', 'mentor', 'teacher', 'manufacturer', \
'coordinator', 'freedom' 'aerospace', 'wireless', 'weekend', 'entertainment', 'investor', 'emergency', \
'water', 'competitor', 'private_medical_insurance', 'human_resource', 'chef', 'employer', 'researcher', \
'problem_solver', 'consult', 'term_condition', 'key_word', 'contributory_pension_scheme', 'birthday', 'airport', \
'administer', 'headquarters', 'corporation', 'analyst_programmer_engineer', \
'professional_registration', 'antivirus', 'central_government', 'public_transport', 'processor', \
'leader_field', 'profession', 'email_address', 'telephone', 'speaker', 'admission', 'bed', \
'psychiatrist', 'microbiologist', 'chinatown', 'security_guard', 'hse_advisor', 'divisional_director', \
'forensic_psychologist', 'marketfactory', 'european_protected_specie', 'protein_scientist', 'circular_economy', 'survivor']

#print(len(no_skills))
# Sono 143 non skills

cor=file_corpus["corpus"]
#cor=cor[-100000:0]
cor=cor.dropna()
#print(type(cor))

#cor=cor[:2]
#print(cor)
#print(cor.head)
#print(len(cor))
#print(type(cor)) #serie di Pandas

# ESTRAZIONE CAMPIONI SKILLS E NON SKILLS PER TRAIN E TEST -----------------------------------------------------------

# Estraggo un campione di skills dalla lista completa importata
n=1000
sample_skills=file_skills.sample(n)
#print(sample_skills)
#print(len(sample_skills))

skills=sample_skills['skills'].tolist()
#print(skills)
#print(len(skills))

# Ora sia skills che no_skills sono liste da 143 elementi

# Per ogni skill e per ogni no skills prendo le 10 parole intorno (nel corpus), i loro vettori e creo una lista
# Ottengo una lista di liste ciascuna con 10 vettori (e 0/1 a seconda che siano intorno ad una skill o ad una noskill)

# --------------------------------------------------------------------------------------------------------------------
# Generazione finestre di parole per skills e non skills

conteggio=0
finestre_skills=[]

for annuncio in cor:
    #print(annuncio)
    parole_annuncio=annuncio.split() #e' una lista di parole
    n_parole=len(parole_annuncio)-1 #indice a cui deve arrivare l'indicatore
    i=0
    while i<=n_parole:
        parola=parole_annuncio[i]
        if parola in skills:
            if conteggio<9500:
                finestra=[] #in finestra ho le 10 o meno parole
                if i-5>=0:
                    parola1=parole_annuncio[i-5]
                    finestra.append(parola1)
                if i-4>=0:
                    parola2=parole_annuncio[i-4]
                    finestra.append(parola2)
                if i-3>=0:
                    parola3=parole_annuncio[i-3]
                    finestra.append(parola3)
                if i-2>=0:
                    parola4=parole_annuncio[i-2]
                    finestra.append(parola4)
                if i-1>=0:
                    parola5=parole_annuncio[i-1]
                    finestra.append(parola5)
                if i+1<n_parole:
                    parola6=parole_annuncio[i+1]
                    finestra.append(parola6)
                if i+2<n_parole:
                    parola7=parole_annuncio[i+2]
                    finestra.append(parola7)
                if i+3<n_parole:
                    parola8=parole_annuncio[i+3]
                    finestra.append(parola8)
                if i+4<n_parole:
                    parola9=parole_annuncio[i+4]
                    finestra.append(parola9)
                if i+5<n_parole:
                    parola10=parole_annuncio[i+5]
                    finestra.append(parola10)

                if i-6>=0:
                    parola11=parole_annuncio[i-6]
                    finestra.append(parola11)
                if i+6<n_parole:
                    parola12=parole_annuncio[i+6]
                    finestra.append(parola12)
                finestre_skills.append(finestra)
                conteggio+=1
        i+=1
#print(finestre_skills)
#print(conteggio)
#print(len(finestre_skills))
# finestre_skills e' una lista di liste

conteggio=0
finestre_noskills=[]

for annuncio in cor:
    parole_annuncio=annuncio.split() #e' una lista di parole
    n_parole=len(parole_annuncio)-1 #indice a cui deve arrivare l'indicatore
    i=0
    while i<=n_parole:
        parola=parole_annuncio[i]
        if parola in no_skills:
            if conteggio<9500:
                finestra=[] #in finestra ho le 10 o meno parole
                if i-5>=0:
                    parola1=parole_annuncio[i-5]
                    finestra.append(parola1)
                if i-4>=0:
                    parola2=parole_annuncio[i-4]
                    finestra.append(parola2)
                if i-3>=0:
                    parola3=parole_annuncio[i-3]
                    finestra.append(parola3)
                if i-2>=0:
                    parola4=parole_annuncio[i-2]
                    finestra.append(parola4)
                if i-1>=0:
                    parola5=parole_annuncio[i-1]
                    finestra.append(parola5)
                if i+1<n_parole:
                    parola6=parole_annuncio[i+1]
                    finestra.append(parola6)
                if i+2<n_parole:
                    parola7=parole_annuncio[i+2]
                    finestra.append(parola7)
                if i+3<n_parole:
                    parola8=parole_annuncio[i+3]
                    finestra.append(parola8)
                if i+4<n_parole:
                    parola9=parole_annuncio[i+4]
                    finestra.append(parola9)
                if i+5<n_parole:
                    parola10=parole_annuncio[i+5]
                    finestra.append(parola10)

                if i-6>=0:
                    parola11=parole_annuncio[i-6]
                    finestra.append(parola11)
                if i+6<n_parole:
                    parola12=parole_annuncio[i+6]
                    finestra.append(parola12)
                finestre_noskills.append(finestra)
                conteggio+=1
        i+=1
#print(finestre_noskills)
#print(conteggio)
#print(len(finestre_noskills))
# finestre_noskills e' una lista di liste

# -------------------------------------------------------------------------------------------------------------------
# Eliminazione finestre uguali:
for lista in finestre_skills:
    finestre_skills_meno1 = [x for x in finestre_skills if x != lista]
    for elemento in finestre_skills_meno1:
        if elemento == lista:
            finestre_skills.remove(elemento)
#print(len(finestre_skills))

for lista in finestre_noskills:
    finestre_noskills_meno1 = [x for x in finestre_noskills if x != lista]
    for elemento in finestre_noskills_meno1:
        if elemento == lista:
            finestre_noskills.remove(elemento)
#print(len(finestre_noskills))

# -----------------------------------------------------------------------------------------------------------------------
# Divisione train e test di skills e non skills
n=6500
skills_train=random.sample(finestre_skills, n)

skills_test=[]
for finestra in finestre_skills:
    if finestra not in skills_train:
        skills_test.append(finestra)

#print(skills_train)
print("Len skills train", len(skills_train))
#print(skills_test)
print("Len skills test", len(skills_test))


noskills_train=random.sample(finestre_noskills, n)

noskills_test=[]
for finestra in finestre_noskills:
    if finestra not in noskills_train:
        noskills_test.append(finestra)

#print(skills_train)
print("Len noskills train", len(noskills_train))
#print(skills_test)
print("Len noskills test", len(noskills_test))

# --------------------------------------------------------------------------------------------------------------------------
# Lista che contiene tutte le parole del training set
lista_train=skills_train+noskills_train
#print(type(lista_train))
#print(lista_train)

# --------------------------------------------------------------------------------------------------------------------------
# Generazione dataset di training e di test e del target y_train e y_test
x_train=[]
y_train=[]
for elemento in skills_train:
    x_train.append(elemento)
    y_train.append(1)
for elemento in noskills_train:
    x_train.append(elemento)
    y_train.append(0)

print(type(x_train)) #lista di liste
print(type(x_train[1]))
print(type(y_train)) #lista

x_test=[]
y_test=[]
for elemento in skills_test:
    x_test.append(elemento)
    y_test.append(1)
for elemento in noskills_test:
    x_test.append(elemento)
    y_test.append(0)

print(type(x_test)) #lista di liste
print(type(x_test[1]))
print(type(y_test)) #lista

# =================================================================================
# CLASSIFICAZIONE

t = Tokenizer()
t.fit_on_texts(lista_train) #cosi' creo il dizionario word_index
vocab_size = len(t.word_index) + 1
emb_size = embeddings.vector_size
#print(vocab_size) #2 col corpus finito 20 righe
#print(emb_size) #100

#print(t.fit_on_texts(x_train)) #e' none
#print(t.word_index) #e' un dizionario
#print(emb_size) #100

max_length = 12
encoded_docs_train = t.texts_to_sequences(x_train) #l'argomento e' una lista di liste di parole
encoded_docs_test = t.texts_to_sequences(x_test)
#print(encoded_docs[1]) #lista di numeri
x_train = pad_sequences(encoded_docs_train, maxlen=max_length, padding='post')
x_test = pad_sequences(encoded_docs_test, maxlen=max_length, padding='post')
#print(padded_docs[1]) #lista di numeri e zeri
#print(len(padded_docs[1])) #1000

# ---------------------------------------

tf.random.set_seed(1)

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, emb_size))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        if word in embeddings:
            weight_matrix[i] = embedding[word]
    return weight_matrix

# get vectors in the right order
embedding_vectors = get_weight_matrix(embeddings, t.word_index)
#print(embedding_vectors.shape) #numerosita' x 100
e = tf.keras.layers.Embedding(vocab_size, emb_size, weights=[embedding_vectors], input_length=max_length, trainable=False)

#params
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
n_labels = 2

# create the model
def baseline_model(optimizer=opt):
    model = tf.keras.Sequential()
    model.add(e)
    model.add(tf.keras.layers.LSTM(100, activation='relu', return_sequences=True)) #in caso mettere 100, prima era 200
    model.add(tf.keras.layers.LSTM(100, activation='relu')) #in caso mettere 100, prima era 200
    model.add(tf.keras.layers.Dense(n_labels, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    return model

model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=baseline_model)
batch_size = [8, 16, 32, 64] #in caso togliere 4
epochs = [10, 20, 50, 100]
optimizer = ['Adam'] #in caso lasciare adam
# batch_size = [64]
# epochs = [50]
# optimizer = ['Adam']
param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer)
#print(param_grid)

# ----------------------------------

#print(type(x_train))
#print(len(y_train))
#print(x_train.info())

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='f1_weighted', verbose=2)
#print(grid)

# print(x_train)
# print(y_train)
# print(type(x_train))
# print(type(x_train[1]))
# print(type(y_train))
# print(type(model))
# print(grid)

# Salvataggio x_train e x_test
train_salvataggio=open('/home/maugeri/ojvs-skills/Paper/my_exp/results/vettori_3sett.csv', 'w')
for elemento in x_train:
    train_salvataggio.write(str(elemento)+'\n')
train_salvataggio.close()

test_salvataggio=open('/home/maugeri/ojvs-skills/Paper/my_exp/results/var_target_23sett.csv', 'w')
for elemento in y_train:
    test_salvataggio.write(str(elemento)+'\n')
test_salvataggio.close()
# 

print('***CROSS VALIDATED GRID SEARCH***')
# print(type(x_train))
# print(type(x_train[1]))
#x_train = np.asarray(x_train).astype("float32")
grid_result = grid.fit(x_train, y_train)
#print(type(x_train[1]))
# print(type(y_train))

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#print(len(y_train))

#results on the test set for the best model
best_model = grid.best_estimator_
y_pred = best_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('***RESULTS ON TEST SET***')
print("accuracy_score", accuracy)
print("f1_score", f1)
print('\n')
print(classification_report(y_test, y_pred))

descrizione_modello_migliore="Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)
print(descrizione_modello_migliore)

# PER SALVARE IN UN FILE .CSV I PARAMETRI DEL MODELLO DI CLASSIFICAZIONE MIGLIORE
file_descrizione = open('/home/maugeri/ojvs-skills/Paper/my_exp/results/descrizione_best_classification_model_23sett.csv', 'w')
file_descrizione.write('Modello di classificazione migliore tra quelli tunati\n\n')
file_descrizione.write(descrizione_modello_migliore+'\n\n')
file_descrizione.write('***RESULTS ON TEST SET***'+'\n')
file_descrizione.write('accuracy score: '+str(accuracy)+'\n')
file_descrizione.write('f1_score: '+str(f1)+'\n\n')
file_descrizione.write('classification report:\n'+str(classification_report(y_test, y_pred)))
file_descrizione.close()

import tensorflow as tf
from tensorflow.keras.models import load_model
# model = best_model.fit(x_train, y_train)
best_model.model.save('/home/maugeri/ojvs-skills/Paper/my_exp/results/modello_classificazione_23sett.h5')

#----------------------------------------------------------------------------------------
'''
#best_model = tf.keras.models.load_model('saved_model/modello_classificazione')
#best_model=load_model('modello_classificazione.h5')

# VALUTAZIONE

#n=2000 #2k annunci sono circa 1 ogni 1000 nel corpus
sample_annunci = cor#.sample(3) # COMMENTARE

tutte_le_parole=[]
for annuncio in sample_annunci:
    parole=annuncio.split()
    tutte_le_parole.extend(parole)

conteggio_parole=collections.Counter(tutte_le_parole)

parole_da_valutare=[]
finestre_parole=[]

for annuncio in sample_annunci:
    parole_annuncio=annuncio.split() #e' una lista di parole
    n_parole=len(parole_annuncio)-1 #indice a cui deve arrivare l'indicatore
    i=0
    while i<=n_parole:
        parola=parole_annuncio[i]
        if parola not in skills and parola not in no_skills and conteggio_parole[parola]>=2000 and conteggio_parole[parola]<=40000: #QUI 1000
            finestra=[] #in finestra ho le 12 o meno parole
            if i-5>=0:
                parola1=parole_annuncio[i-5]
                finestra.append(parola1)
            if i-4>=0:
                parola2=parole_annuncio[i-4]
                finestra.append(parola2)
            if i-3>=0:
                parola3=parole_annuncio[i-3]
                finestra.append(parola3)
            if i-2>=0:
                parola4=parole_annuncio[i-2]
                finestra.append(parola4)
            if i-1>=0:
                parola5=parole_annuncio[i-1]
                finestra.append(parola5)
            if i+1<n_parole:
                parola6=parole_annuncio[i+1]
                finestra.append(parola6)
            if i+2<n_parole:
                parola7=parole_annuncio[i+2]
                finestra.append(parola7)
            if i+3<n_parole:
                parola8=parole_annuncio[i+3]
                finestra.append(parola8)
            if i+4<n_parole:
                parola9=parole_annuncio[i+4]
                finestra.append(parola9)
            if i+5<n_parole:
                parola10=parole_annuncio[i+5]
                finestra.append(parola10)

            if i-6>=0:
                parola11=parole_annuncio[i-6]
                finestra.append(parola11)
            if i+6<n_parole:
                parola12=parole_annuncio[i+6]
                finestra.append(parola12)
            parole_da_valutare.append(parola)
            finestre_parole.append(finestra)
        i+=1
#print(parole_da_valutare)
#print(len(parole_da_valutare))
#print(finestre_parole[:5])
#print(len(finestre_parole))
parole_da_valutare=parole_da_valutare[:20000]
finestre_parole=finestre_parole[:20000]

file_n_finestre=open('/home/maugeri/ojvs-skills/Paper/my_exp/results/numero_finestre_21sett.csv', 'w')
file_n_finestre.write('numero di finestre valutate: ' + str(len(parole_da_valutare)))
file_n_finestre.close()

# counter_parole=collections.Counter(parole_da_valutare)
# for parola, finestra in zip(parole_da_valutare, finestre_parole):
#     if counter_parole[parola]<1000: #rimuovo le parole dalla valutazione se appaiono meno di 200 volte nel corpus
#         parole_da_valutare.remove(parola)
#         finestre_parole.remove(finestra)

max_length=12
t = Tokenizer()
t.fit_on_texts(finestre_parole) #cosi' creo il dizionario word_index
encoded_docs_valutazione = t.texts_to_sequences(finestre_parole) #l'argomento e' una lista di liste
#print(encoded_docs_valutazione[:5])
x_valutazione = pad_sequences(encoded_docs_valutazione, maxlen=max_length, padding='post')

file_x_val=open('/home/maugeri/ojvs-skills/Paper/my_exp/results/parola_finestra_vettore_21sett.csv', 'w')
for parola, finestra, vettore in zip(parole_da_valutare, finestre_parole, x_valutazione):
    contenuto=(parola, finestra, vettore)
    file_x_val.write(str(contenuto))
file_x_val.close()


y_valutazione = best_model.predict_proba(x_valutazione) #contiene le probabilità
print(y_valutazione[:5])
#print(len(y_valutazione))
file_y_val=open('/home/maugeri/ojvs-skills/Paper/my_exp/results/parola_yval_21sett.py', 'w')
file_y_val.write(str(y_valutazione))
file_y_val.close()

# ----------------------------------------------------------------------------------------------

parole_per_counter=[]
parole_prob=[]
for parola, prob in zip(parole_da_valutare, y_valutazione):
    probabilità=prob[1]
    if probabilità > 0.5:
        parole_per_counter.append(parola)
        coppia=(parola, probabilità)
        parole_prob.append(coppia)
print(parole_per_counter)
print(parole_prob)

dizionario_counter = collections.Counter(parole_per_counter)
print(dizionario_counter)

dizionario_prob = defaultdict(list)
for key, val in parole_prob:
    dizionario_prob[key].append(val)
print(dizionario_prob)

# FILE1 CONTIENE PAROLA, CONTEGGIO, PROABILITA' MASSIMA E PROBABILITA' MEDIA

file1_cont=[]
for parola in dizionario_prob:
    tupla=(parola, dizionario_counter[parola], max(dizionario_prob[parola]), np.mean(dizionario_prob[parola]))
    file1_cont.append(tupla)
print(file1_cont)

file1 = open('/home/maugeri/ojvs-skills/Paper/my_exp/results/file1_valutazione_21sett.csv', 'w') #QUI DATA
file1.write('Parola, conteggio, probabilità di classificazione maggiore, probabilità media di classificazione\n')
for elemento in file1_cont:
    file1.write(str(elemento)+'\n')
file1.close()

# FILE2 CONTIENE PAROLA, CONTEGGIO, SINGOLE PROBABILITA' E PROBABILITA' MEDIA

file2_cont=[]
for parola in parole_prob:
    tupla=(parola[0], dizionario_counter[parola[0]], parola[1], np.mean(dizionario_prob[parola[0]]))
    file2_cont.append(tupla)
print(file2_cont)

file2 = open('/home/maugeri/ojvs-skills/Paper/my_exp/results/file2_valutazione_21sett.csv', 'w') #QUI DATA
file2.write('Parola, conteggio, probabilità di classificazione, probabilità media di classificazione\n')
for elemento in file2_cont:
    file2.write(str(elemento)+'\n')
file2.close()

# file=open('valutazione_new_skills_2m_40k_5k.csv', 'w')
# file.write("Valutazione nuove skills\n")
# parole=[]
# for parola, target_previsto in zip(parole_da_valutare, y_valutazione):
#     if target_previsto == 1 and parola not in parole:
#         file.write(str(parola)+'\n')
#         parole.append(parola)
# file.close()

#----------------------------------------------------------------------------------
# COUNTERS PER PAROLE CHE APPAIONO ALMENO 200 O 500 VOLTE

file_parole_almeno_500 = open('/home/maugeri/ojvs-skills/Paper/my_exp/results/file_parole_almeno_500_21sett.csv', 'w')
file_parole_almeno_500.write('Parole che appaiono almeno 500 volte, conteggio')
for key in dizionario_counter:
    if dizionario_counter[key]>=500:
        file_parole_almeno_500.write(str(key)+' '+str(dizionario_counter[key])+'\n')
file_parole_almeno_500.close()

file_parole_almeno_300 = open('/home/maugeri/ojvs-skills/Paper/my_exp/results/file_parole_almeno_300_21sett.csv', 'w')
file_parole_almeno_300.write('Parole che appaiono almeno 300 volte, conteggio')
for key in dizionario_counter:
    if dizionario_counter[key]>=300:
        file_parole_almeno_300.write(str(key)+' '+str(dizionario_counter[key])+'\n')
file_parole_almeno_300.close()
'''