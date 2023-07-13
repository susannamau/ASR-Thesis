import pickle
import numpy as np

with open('/home/maugeri/ojvs-skills/Paper/JD2Skills-BERT-XMLC-main/pybert/dataset/job_dataset.train.pkl', 'rb') as file:
    train = pickle.load(file)
#print(len(train[0][1]))
#print(train[63:65])
'''
#print(train[1][0][60:70])

for element in train:
    stringa= element[0]
    if isinstance(stringa, str):
        print(type(stringa))


with open('/home/maugeri/ojvs-skills/Paper/JD2Skills-BERT-XMLC-main/pybert/dataset/job_dataset.test.pkl', 'rb') as file:
    test = pickle.load(file)

for element in test:
    stringa= element[0][60:70]
    if not isinstance(stringa, str):
        print(stringa)

with open('/home/maugeri/ojvs-skills/Paper/JD2Skills-BERT-XMLC-main/pybert/dataset/job_dataset.valid.pkl', 'rb') as file:
    valid = pickle.load(file)

for element in valid:
    stringa= element[0][60:70]
    if not isinstance(stringa, str):
        print(stringa)
'''

# with gzip.open('job_dataset.train.pkl', 'rb') as ifp:
#     print(pickle.load(ifp))