#!/usr/bin/python3

# IMPORTANT: EXECUTAR AMB VERSIÓ 3 DE PYTHON

# Definició de imports i froms
import constants
import os, os.path
import nltk
import string
import codecs
import math
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

#Definició de varibales globals
mailDir  = "./mails/ENTRENAMENT"
validacioDir  = "./mails/VALIDACIO"
mails_ham = []
mails_spam = []
nombre_paraules_correus = 0
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
    #signes_puntuacio = set(string.punctuation + '')

    clean_string = [token for token in llista_paraules_lower if not token in stop_words]
    clean_string = [token for token in clean_string if not token.isnumeric()]
    clean_string = [eliminar_puntuacio(token) for token in clean_string]
    clean_string = list(filter(None, clean_string))

    return clean_string    

def calcular_probabilitat(word, frequencia_paraules,vocTotal,N):
    # p(x) = (count+K)/(N+(K*|x|))
    prob = (frequencia_paraules[word] + constants.K)/(N+(constants.K*vocTotal))
    return prob

def metode_Bayes( conjunt_paraules_correu, frequencia_paraules_ham, frequencia_paraules_spam, mida_vocabulari, NParaulesHam, NParaulesSpam):
    aux_spam = 0
    aux_ham = 0

    for word in conjunt_paraules_correu:
        aux_spam += math.log(calcular_probabilitat(word,frequencia_paraules_spam, mida_vocabulari, NParaulesSpam))
        aux_ham += math.log(calcular_probabilitat(word,frequencia_paraules_ham, mida_vocabulari, NParaulesHam))
    aux_spam += math.log(probabilitat_spam())
    aux_ham += math.log(probabilitat_ham())

    prob_spam = aux_spam/(aux_ham+aux_spam)
    prob_ham = aux_ham/(aux_ham+aux_spam)
    return prob_spam > (prob_ham*constants.PHI)


def probabilitat_spam():
    return ((n_missatges_spam + constants.K)/(nombre_paraules_correus + (constants.K*2)))
    
def probabilitat_ham():
    return ((n_missatges_ham + constants.K)/(nombre_paraules_correus + (constants.K*2)))

def obtenir_paraules_correu(filename):
    return (codecs.open(filename, "rb", "latin-1").read().split())

def calcular_estadistics(n_missatges_ham, n_missatges_spam, true_positiu, fals_positiu, true_negatiu, fals_negatiu):
    missatges_totals = (n_missatges_ham + n_missatges_spam)
    accuracy = 0.0
    fals_positiu_rate = (fals_positiu/missatges_totals)*100
    fals_negatiu_rate = (fals_negatiu/missatges_totals)*100
    tct = total_cost_ratio(fals_positiu, fals_negatiu, n_missatges_ham, n_missatges_spam)
    print ("-------------------- RESULTATS ---------------------")
    print ("Nombre de missatges: \t" + str(missatges_totals) + "\t (" + str(n_missatges_ham) + "H," + str(n_missatges_spam) + "S)")
    print ("Accuracy (%): \t " + str(accuracy))
    print ("False positive rate (%): \t" + str(fals_positiu_rate) + "\t (" + str(fals_positiu) + ")")
    print ("False negative rate (%): \t" + str(fals_negatiu_rate) + "\t (" + str(fals_negatiu) + ")")
    print ("Total cost ratio (l = 50): \t" + str(tct))
    print ("----------------------------------------------------")

def main():
    # Lectura de fitxers del directori mailDir
    for directory, subdirs, files in os.walk(mailDir):
        for filename in files:
            filepath = os.path.join(directory, filename)
            if "SPAM" in filename.upper():
                mails_spam.extend(obtenir_paraules_correu(filepath))
            elif "HAM" in filename.upper():
                mails_ham.extend(obtenir_paraules_correu(filepath))

    llista_paraules_neta_ham = neteja_paraules(mails_ham) #totes les paraules que apareixen a ham
    llista_paraules_neta_spam = neteja_paraules(mails_spam) #totes les paraules que apareixen a spam


    frequencia_paraules_ham = nltk.FreqDist(llista_paraules_neta_ham) # mapa de les paraules i la seva frequencia HAM
    frequencia_paraules_spam = nltk.FreqDist(llista_paraules_neta_spam) # mapa de les paraules i la seva frequencia SPAM
    n_missatges_ham = len(llista_paraules_neta_ham) 
    n_missatges_spam = len(llista_paraules_neta_spam) 
    nombre_paraules_correus = n_missatges_ham + n_missatges_spam # nombre total de paraules no rep

    mida_vocabulari = len(frequencia_paraules_spam.items())+len(frequencia_paraules_ham.items()) #mida total del vocabulari

    #for key,val in frequencia_paraules_spam.items():
        #print (str(key) + ' : ' + str(val))

    # print ("")
    # print ("Nombre de paruales correus ==> " + str(nombre_paraules_correus))
    # print ("Mida vocabulari = " + str(mida_vocabulari))
    # print ("missatges SPAM = " + str(n_missatges_spam))
    # print ("missatges HAM = " + str(n_missatges_ham))
    # print ("missatges TOTALS = " + str(nombre_paraules_correus))
    # print ("PROBABILITAT SPAM = " + str(probabilitat_spam()))
    # print ("PROBABILITAT HAM = " + str(probabilitat_ham()))

    # print (llista_paraules_neta_ham)
    # for key,val in frequencia_paraules.items():
    #     print (str(key) + ' : ' + str(val))

    # Per cada missatge de la carpeta validacioDir...
    validacio_paraules_spam = []
    validacio_paraules_ham = []
    es_spam = False

    n_missatges_ham_validacio = 0
    n_missatges_spam_validacio = 0
    true_positiu = 0
    false_positiu = 0
    true_negatiu = 0
    false_negatiu = 0

    for directory, subdirs, files in os.walk(validacioDir):
        for filename in files:
            filepath = os.path.join(directory, filename)
            if "SPAM" in filename.upper():
                n_missatges_spam_validacio+=1
                validacio_paraules_spam = neteja_paraules(obtenir_paraules_correu(filepath))
                es_spam = True
                bayes = metode_Bayes(validacio_paraules_spam, frequencia_paraules_ham, frequencia_paraules_spam, mida_vocabulari, n_missatges_ham, n_missatges_spam)
            else :
                n_missatges_ham_validacio+=1
                validacio_paraules_ham = neteja_paraules(obtenir_paraules_correu(filepath))
                bayes = metode_Bayes(validacio_paraules_spam, frequencia_paraules_ham, frequencia_paraules_spam, mida_vocabulari, n_missatges_ham, n_missatges_spam)
            
            if es_spam and bayes:
                true_positiu+=1
            elif not es_spam and bayes:
                false_positiu+=1
            elif es_spam and not bayes:
                true_negatiu+=1
            else:
                false_negatiu+=1

    calcular_estadistics(n_missatges_ham_validacio, n_missatges_spam_validacio, true_positiu, false_positiu, true_negatiu, false_negatiu)


            

    #print total_cost_ratio(9.0,688.0,29443.0,27220.0)
    #print(', '.join(mails))
    #print(len(mails))


    # |X| -> mida del vocabulari
    # count(xi) -> nombre de ocurrencies d'una paraula
    # N -> nombre de paraules entre tots els correus
    # N -> nombre de paraules de tot ham o spam

main()