#!/usr/bin/python3

# IMPORTANT: EXECUTAR AMB VERSIÓ 3 DE PYTHON

# Definició de imports i froms
import constants
import os, os.path
import nltk
import string
import codecs
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

#Definició de varibales globals
mailDir  = "./mails/ENTRENAMENT"
mails_ham = []
mails_spam = []
n_missatges = 0
n_missatges_spam = 0
n_missatges_ham = 0

def total_cost_ratio(false_positive, false_negative, n_ham, n_spam):
    lambda_value = 50+.0
    werr = (lambda_value * false_positive + false_negative) / (lambda_value * n_ham + n_spam)
    werr_base = n_spam / (lambda_value * n_ham + n_spam)
    return werr_base / werr

def eliminar_puntuacio(string_eliminar):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in string_eliminar if ch not in exclude)

def neteja_paraules(llista_paraules):
    llista_paraules_lower = [token.lower() for token in llista_paraules]
    clean_string = llista_paraules_lower
    stop_words = set(stopwords.words('english'))
    signes_puntuacio = set(string.punctuation + '')

    clean_string = [token for token in llista_paraules_lower if not token in stop_words]
    clean_string = [token for token in clean_string if not token.isnumeric()]
    clean_string = [eliminar_puntuacio(token) for token in clean_string]
    clean_string = list(filter(None, clean_string))

    return clean_string    

def generar_bagofwords(sentence, words):
    sentence_words = neteja_paraules(sentence.split())
    # frequency word count
    bag = np.zeros(len(words))
    for sw in sentence_words:
        for i,word in enumerate(words):
            if word == sw: 
                bag[i] += 1

<<<<<<< HEAD
def calcular_probabilitat(word, llista_paraules,N,K):
    # p(x) = (count+K)/(N+(K*|x|))
    prob = (llista_paraules[word] + K)/(N+(K*len(llista_paraules.items())))
    return prob
=======
    return np.array(bag)        

def probabilitat_spam():
    return ((n_missatges_spam + constants.K)/(n_missatges + (constants.K*2)))
    
def probabilitat_ham():
    return ((n_missatges_ham + constants.K)/(n_missatges + (constants.K*2)))    
>>>>>>> df375fea36ed8f75e5fdd3306f2c100c15bf9fa7

# Lectura de fitxers del directori mailDir
for directory, subdirs, files in os.walk(mailDir):
    for filename in files:
        filepath = os.path.join(directory, filename)
        if "SPAM" in filename.upper():
            mails_spam.extend(codecs.open(filepath, "rb", "latin-1").read().split())
        elif "HAM" in filename.upper():
            mails_ham.extend(codecs.open(filepath, "rb", "latin-1").read().split())


llista_paraules_neta_ham = neteja_paraules(mails_ham)
llista_paraules_neta_spam = neteja_paraules(mails_spam)
frequencia_paraules = nltk.FreqDist(llista_paraules_neta_ham)
n_missatges_ham = len(llista_paraules_neta_ham)
n_missatges_spam = len(llista_paraules_neta_spam)
n_missatges = len(llista_paraules_neta_ham) + len(llista_paraules_neta_spam)

#bag_of_words = generar_bagofwords("Subject, Subject, Subject hola day, people enron enron hola hola", list(frequencia_paraules.keys()))


<<<<<<< HEAD
llista_paraules_neta = neteja_paraules(mails)
nombre_paraules_correus = len( llista_paraules_neta)
frequencia_paraules = nltk.FreqDist(llista_paraules_neta)
mida_vocabulari = len(frequencia_paraules.items())
for key,val in frequencia_paraules.most_common(100):
    print (str(key) + ' : ' + str(val))

print ("")
print ("Nombre de paruales correus ==> " + str(nombre_paraules_correus))
print ("Mida vocabulari = " + str(mida_vocabulari))
=======
print ("missatges SPAM = " + str(n_missatges_spam))
print ("missatges HAM = " + str(n_missatges_ham))
print ("missatges TOTALS = " + str(n_missatges))
print ("PROBABILITAT SPAM = " + str(probabilitat_spam()))
print ("PROBABILITAT HAM = " + str(probabilitat_ham()))

# print (llista_paraules_neta_ham)
# for key,val in frequencia_paraules.items():
#     print (str(key) + ' : ' + str(val))

>>>>>>> df375fea36ed8f75e5fdd3306f2c100c15bf9fa7

prob = calcular_probabilitat("office",frequencia_paraules, nombre_paraules_correus,1)
print(str(prob))

#print total_cost_ratio(9.0,688.0,29443.0,27220.0)
#print(', '.join(mails))
#print(len(mails))


# |X| -> mida del vocabulari
# count(xi) -> nombre de ocurrencies d'una paraula
# N -> nombre de paraules entre tots els correus