#!/usr/bin/python3

# IMPORTANT: EXECUTAR AMB VERSIÓ 3 DE PYTHON
import os
import nltk
import string
import codecs

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

def total_cost_ratio(false_positive, false_negative, n_ham, n_spam):
    lambda_value = 50+.0
    werr = (lambda_value * false_positive + false_negative) / (lambda_value * n_ham + n_spam)
    werr_base = n_spam / (lambda_value * n_ham + n_spam)
    return werr_base / werr

def neteja_paraules(llista_paraules):
    llista_paraules_lower = [token.lower() for token in llista_paraules]
    clean_string = llista_paraules_lower
    stop_words = set(stopwords.words('english'))
    signes_puntuacio = string.punctuation

    clean_string = [token for token in llista_paraules_lower if not token in stop_words]
    clean_string = [token.translate(signes_puntuacio) for token in clean_string if not token in signes_puntuacio]
    clean_string = [token for token in clean_string if not token.isnumeric()]

    return clean_string        

def calcular_probabilitat(word, llista_paraules,N,K):
    # p(x) = (count+K)/(N+(K*|x|))
    prob = (llista_paraules[word] + K)/(N+(K*len(llista_paraules.items())))
    return prob

# Lectura de fitxers del directori mailDir
mailDir  = "./mails/SPAM"
mails = []
mida_vocabulari = 0
nombre_paraules_correus = 0
for directory, subdirs, files in os.walk(mailDir):
    for filename in files:
        filepath = os.path.join(directory, filename)
        mails.extend(codecs.open(filepath, "rb", "latin-1").read().split())


llista_paraules_neta = neteja_paraules(mails)
nombre_paraules_correus = len( llista_paraules_neta)
frequencia_paraules = nltk.FreqDist(llista_paraules_neta)
mida_vocabulari = len(frequencia_paraules.items())
for key,val in frequencia_paraules.most_common(100):
    print (str(key) + ' : ' + str(val))

print ("")
print ("Nombre de paruales correus ==> " + str(nombre_paraules_correus))
print ("Mida vocabulari = " + str(mida_vocabulari))

prob = calcular_probabilitat("office",frequencia_paraules, nombre_paraules_correus,1)
print(str(prob))

#print total_cost_ratio(9.0,688.0,29443.0,27220.0)
#print(', '.join(mails))
#print(len(mails))


# |X| -> mida del vocabulari
# count(xi) -> nombre de ocurrencies d'una paraula
# N -> nombre de paraules entre tots els correus