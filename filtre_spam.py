import os
import nltk
import string

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
    clean_string = [token.translate(None, signes_puntuacio) for token in clean_string if not token in signes_puntuacio]

    return clean_string        


mailDir  = "./mails/HAM"

mails = []
for directory, subdirs, files in os.walk(mailDir):
    for filename in files:
        filepath = os.path.join(directory, filename)
        mails.extend(open(filepath, "rb").read().split())


llista_paraules_neta = neteja_paraules(mails)

freq = nltk.FreqDist(llista_paraules_neta)
for key,val in freq.items():
    print (str(key) + ' : ' + str(val))

print ("Mida llista paraules neta ==> " + str(len( llista_paraules_neta)))

#print total_cost_ratio(9.0,688.0,29443.0,27220.0)
#print(', '.join(mails))
#print(len(mails))


# |X| -> mida del vocabulari
# count(xi) -> nombre de ocurrencies d'una paraula
# N -> nombre de paraules entre tots els correus