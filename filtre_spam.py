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
import io

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

#Definició de varibales globals
mailDir  = "./mails/ENTRENAMENT"
validacioDir  = "./mails/VALIDACIO"

#calcula el costRatio a partir del fals positiu, el fals negatiu, el nombre de missatges de ham i spam
def total_cost_ratio(false_positive, false_negative, n_ham, n_spam):
    lambda_value = 50.0
    werr = (lambda_value * false_positive + false_negative) / (lambda_value * n_ham + n_spam)
    if(werr == 0):
        werr = 0.000001    
    werr_base = n_spam / (lambda_value * n_ham + n_spam)
    return (werr_base / werr)

#elimina els signes de puntuació de l'string
def eliminar_puntuacio(string_eliminar):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in string_eliminar if ch not in exclude)

#neteja de llista_paraules els simbols de puntuacio, els números, els tokens (anglesos)
def neteja_paraules(llista_paraules):
    llista_paraules_lower = [token.lower() for token in llista_paraules]
    clean_string = llista_paraules_lower
    stop_words = set(stopwords.words('english'))
    st = WordNetLemmatizer()

    clean_string = [token for token in llista_paraules_lower if not token in stop_words]
    clean_string = [token for token in clean_string if not token.isnumeric()]
    clean_string = [eliminar_puntuacio(token) for token in clean_string]
    clean_string = list(filter(None, clean_string))
    #clean_string = [st.lemmatize(token) for token in clean_string]
    
    return clean_string    

#calcula la probabilitat de que word sigui SPAM o HAM depenent dels parametres que li passem
def calcular_probabilitat(word, frequencia_paraules,vocTotal,N):
    prob = (frequencia_paraules[word] + constants.K)/(N+(constants.K*vocTotal))
    return prob

#retorna cert si a partir de Bayes el conjunt_paraules_correu calculem que es SPAM, fals si és HAM
def metode_Bayes( conjunt_paraules_correu, frequencia_paraules_ham, frequencia_paraules_spam, mida_vocabulari, NParaulesHam, NParaulesSpam):
    prob_spam = 0
    prob_ham = 0
    for word in conjunt_paraules_correu:
        prob_spam += math.log(calcular_probabilitat(word,frequencia_paraules_spam, mida_vocabulari, NParaulesSpam))
        prob_ham += math.log(calcular_probabilitat(word,frequencia_paraules_ham, mida_vocabulari, NParaulesHam))
    return (prob_spam > (prob_ham + math.log(constants.PHI)))

#mentre li passem el mateix nombre de correus de HAM que d'SPAM, sempre serà 0.5
def probabilitat_spam():
    return 0.5
def probabilitat_ham():
    return 0.5

def obtenir_paraules_correu(filename):
    return codecs.open(filename, 'r', encoding='utf-8', errors='ignore').read().split()

def calcular_estadistics(n_missatges_ham, n_missatges_spam, true_positiu, fals_positiu, true_negatiu, fals_negatiu):
    missatges_totals = (n_missatges_ham + n_missatges_spam)
    bones_prediccions = missatges_totals - (fals_negatiu + fals_positiu)
    accuracy = (bones_prediccions/missatges_totals)*100
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
    mails_ham = []
    mails_spam = []
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
    frequencia_paraules_totals = nltk.FreqDist(llista_paraules_neta_ham+llista_paraules_neta_spam)
    n_missatges_ham = len(llista_paraules_neta_ham) 
    n_missatges_spam = len(llista_paraules_neta_spam) 
    mida_vocabulari = len(frequencia_paraules_totals.items()) #mida total del vocabulari

    # Per cada missatge de la carpeta validacioDir...
    validacio_paraules_spam = []
    validacio_paraules_ham = []
    es_spam = False

    n_missatges_ham_validacio = 0
    n_missatges_spam_validacio = 0
    true_positiu = 0 #detces spam
    false_positiu = 0 #quan et ve un ham i el detectes com spam
    true_negatiu = 0 #detes ham
    false_negatiu = 0 #quan et ve un spam i no el detectes

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
                bayes = metode_Bayes(validacio_paraules_ham, frequencia_paraules_ham, frequencia_paraules_spam, mida_vocabulari, n_missatges_ham, n_missatges_spam)
            
            if es_spam and bayes:
                true_positiu+=1
            elif not es_spam and bayes:
                false_positiu+=1
            elif es_spam and not bayes:
                false_negatiu+=1
            elif not es_spam and not bayes:
                true_negatiu+=1

    print("VALOR DE K => ", constants.K, " VALOR DE PHI => ", constants.PHI)         
    calcular_estadistics(n_missatges_ham_validacio, n_missatges_spam_validacio, true_positiu, false_positiu, true_negatiu, false_negatiu)

main()