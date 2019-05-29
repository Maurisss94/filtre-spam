import nltk
import os
import random
import codecs
import nltk.classify.util
from collections import Counter
from nltk import word_tokenize, WordNetLemmatizer
from nltk import NaiveBayesClassifier, classify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stoplist = stopwords.words('english')

def train(features, all_features_valid):
    # initialise the training and test sets
    train_set, test_set = features, all_features_valid
    print ('Training set of size= ' + str(len(train_set)) + ' mails')
    print ('Test set of size = ' + str(len(test_set)) + ' mails')
    # train the classifier
    classifier = NaiveBayesClassifier.train(train_set)
    return test_set, classifier

def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(str(sentence))]

def get_features(text, setting):
    if setting=='bow':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    else:
        return {word: True for word in preprocess(text) if not word in stoplist}

def init_lists(folder):
    key_list = []
    file_content = os.listdir(folder)
    for a_file in file_content:
        filepath = os.path.join(folder, a_file)
        f = codecs.open(filepath, 'r', encoding='utf-8', errors='ignore')
        key_list.append(f.read())
    f.close()
    return key_list
    
def evaluate(test_set, classifier):
    print ('Test set accuracy = ' + str(classify.accuracy(classifier, test_set)))
    # check most informative words for the classifier
    classifier.show_most_informative_features(20)


spam = init_lists("./mails/ENTRENAMENT/SPAM/")
ham = init_lists("./mails/ENTRENAMENT/HAM/")
validacioSpam  = init_lists("./mails/emailscampionat/SPAM/")
validacioHam  = init_lists("./mails/emailscampionat/HAM/")
all_mails = [(mail, 'spam') for mail in spam]
all_mails += [(mail, 'ham') for mail in ham]
all_mails_valid = [(mail, 'spam') for mail in validacioSpam]
all_mails_valid += [(mail, 'ham') for mail in validacioHam]
random.shuffle(all_mails)
random.shuffle(all_mails_valid)
print ('Corpus of size = ' + str(len(all_mails)) + ' mails')

all_features = [(get_features(mail, ''), label) for (mail, label) in all_mails]
all_features_valid = [(get_features(mail, ''), label) for (mail, label) in all_mails_valid]
print ('Fetched ' + str(len(all_features)) + ' feature sets')

test_set, classifier = train(all_features, all_features_valid)

evaluate(test_set, classifier)