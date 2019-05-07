import os
import nltk
import string
from nltk.corpus import stopwords

def total_cost_ratio(false_positive, false_negative, n_ham, n_spam):
    lambda_value = 50+.0
    werr = (lambda_value * false_positive + false_negative) / (lambda_value * n_ham + n_spam)
    werr_base = n_spam / (lambda_value * n_ham + n_spam)
    return werr_base / werr

mailDir  = "./mails/PROVA"

mails = []
mails2 = []
for directory, subdirs, files in os.walk(mailDir):
    for filename in files:
        filepath = os.path.join(directory, filename)
        mails2.extend(open(filepath, "rb").read().split())



clean_tokens = mails2
sr = stopwords.words('english')



for s in mails2:
    mails.append(s.translate(None, string.punctuation))

# for token in mails:
#     if token not in sr and token not in string.punctuation and token not in string.digits:
#         #print(token)
#         clean_tokens.append(token)

# |X| -> mida del vocabulari
# count(xi) -> nombre de ocurrencies d'una paraula
# N -> nombre de paraules entre tots els correus

for token in mails:
    if token in sr and token in string.punctuation and token  in string.digits:
        #print(token)
        clean_tokens.remove(token)

freq = nltk.FreqDist(clean_tokens)
for key,val in freq.items():
    print (str(key) + ':' + str(val))

#print total_cost_ratio(9.0,688.0,29443.0,27220.0)
#print(', '.join(mails))
#print(len(mails))