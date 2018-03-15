#### This time, we're using a movie reviews data set that contains much shorter movie reviews. 
#### This one yields us a far more reliable reading across the board, and is far more fitting 
#### for the tweets we intend to read from the Twitter API soon

import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers= classifiers
    
    def classify(self, features):
        votes= []
        for c in self._classifiers:
            v= c.classify(features)
            votes.append(v)
        try:    
            return mode(votes)
        except Exception as e:
            print(str(e))
    
    def confidence(self, features):
        votes= []
        for c in self._classifiers:
            v= c.classify(features)
            votes.append(v)
        choice_votes= votes.count(mode(votes))
        conf= choice_votes/ len(votes)
        return conf

short_pos= open("short_reviews\\positive.txt", "r").read()
short_neg= open("short_reviews\\negative.txt", "r").read()

all_words_f= open('short_reviews\\all_words.pickle', 'rb')
all_words=pickle.load(all_words_f)
all_words_f.close()

documents_f= open('short_reviews\documents.pickle', 'rb')
documents= pickle.load(documents_f)
documents_f.close()

'''
allowed_word_types= ["J"]  # J for adverb
for r in short_pos.split('\n'):
    #documents.append((r, "pos"))
    words= word_tokenize(r)
    pos= nltk.pos_tag(words)
    for w in pos:
        if(w[1][0] in allowed_word_types):
            all_words.append(w[0].lower())
    
for r in short_neg.split('\n'):
    #documents.append((r, "neg"))
    words= word_tokenize(r)
    pos= nltk.pos_tag(words)
    for w in pos:
        if(w[1][0] in allowed_word_types):
            all_words.append(w[0].lower())
'''

'''
# save the documents
documents_f= open('short_reviews\documents.pickle', "wb")  # \ will be used when '' is used (char)
pickle.dump(documents , documents_f)
documents_f.close()
'''   

all_words = nltk.FreqDist(all_words)  # converting list of words to frequency distribution dictionary

'''
all_words_f= open("short_reviews\\all_words.pickle", "wb")
pickle.dump(all_words, all_words_f)
all_words_f.close()
'''

# taking only top 5000 words as features
word_features = [w[0] for w in all_words.most_common(5000)]

# function to find words which are present in a document
def find_features(document):
    words= set(word_tokenize(document))   # this is all unique words in the document
    features = {}
    for w in word_features:
        features[w] = (w in words)  # boolean of whether w in present in the document
        
    return features 

featureset = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featureset)

training_set = featureset[:10000]
testing_set = featureset[10000:]

'''
classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier.train(training_set)
print("classifier Accuracy is : ", (nltk.classify.accuracy(classifier, testing_set))*100)
#save the classifier
classifier_f = open("short_reviews\\naivebayes.pickle", 'wb')
pickle.dump(classifier, classifier_f)
classifier_f.close()
'''
classifier_f= open("short_reviews\\SVC.pickle", 'rb')
classifier= pickle.load(classifier_f)
classifier_f.close()
print("Original Naive bayes Accuracy is : ", (nltk.classify.accuracy(classifier, testing_set))*100)
#classifier.show_most_informative_features(15)

'''
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier Accuracy is : ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
#save the MNB_classifier
MNB_classifier_f = open("short_reviews\\MultinomialNB.pickle", 'wb')
pickle.dump(MNB_classifier, MNB_classifier_f)
MNB_classifier_f.close()
'''
MNB_classifier_f= open("short_reviews\\MultinomialNB.pickle", "rb")
MNB_classifier= pickle.load(MNB_classifier_f)
MNB_classifier_f.close()
print("MNB_classifier Accuracy is : ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

'''
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BNB_classifier Accuracy is : ", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)
#save the BNB_classifier
BNB_classifier_f = open("short_reviews\\BinomialNB.pickle", 'wb')
pickle.dump(BNB_classifier, BNB_classifier_f)
BNB_classifier_f.close()
'''
BNB_classifier_f= open("short_reviews\\BinomialNB.pickle", "rb")
BNB_classifier= pickle.load(BNB_classifier_f)
BNB_classifier_f.close()
print("BNB_classifier Accuracy is : ", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)

'''
LR_classifier = SklearnClassifier(LogisticRegression())
LR_classifier.train(training_set)
print("LR_classifier Accuracy is : ", (nltk.classify.accuracy(LR_classifier, testing_set))*100)
#save the LR_classifier
LR_classifier_f = open("short_reviews\\LogisticRegression.pickle", 'wb')
pickle.dump(LR_classifier, LR_classifier_f)
LR_classifier_f.close()
'''
LR_classifier_f= open("short_reviews\\LogisticRegression.pickle", "rb")
LR_classifier= pickle.load(LR_classifier_f)
LR_classifier_f.close()
print("LR_classifier Accuracy is : ", (nltk.classify.accuracy(LR_classifier, testing_set))*100)

'''
SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print("SGD_classifier Accuracy is : ", (nltk.classify.accuracy(SGD_classifier, testing_set))*100)
#save the SGD_classifier
SGD_classifier_f = open("short_reviews\\StochasticGD.pickle", 'wb')
pickle.dump(SGD_classifier, SGD_classifier_f)
SGD_classifier_f.close()
'''
SGD_classifier_f= open("short_reviews\\StochasticGD.pickle", "rb")
SGD_classifier= pickle.load(SGD_classifier_f)
SGD_classifier_f.close()
print("SGD_classifier Accuracy is : ", (nltk.classify.accuracy(SGD_classifier, testing_set))*100)

'''
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier Accuracy is : ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)
#save the SVC_classifier
SVC_classifier_f = open("short_reviews\\SVC.pickle", 'wb')
pickle.dump(SVC_classifier, SVC_classifier_f)
SVC_classifier_f.close()
'''
SVC_classifier_f= open("short_reviews\\SVC.pickle", "rb")
SVC_classifier= pickle.load(SVC_classifier_f)
SVC_classifier_f.close()
print("SVC_classifier Accuracy is : ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

'''
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier Accuracy is : ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
#save the LinearSVC_classifier
LinearSVC_classifier_f = open("short_reviews\\LinearSVC.pickle", 'wb')   # \\ will be used when "" are used (string)
pickle.dump(LinearSVC_classifier, LinearSVC_classifier_f)
LinearSVC_classifier_f.close()
'''
LinearSVC_classifier_f= open("short_reviews\\LinearSVC.pickle", "rb")
LinearSVC_classifier= pickle.load(LinearSVC_classifier_f)
LinearSVC_classifier_f.close()
print("LinearSVC_classifier Accuracy is : ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)


voted_classifier= VoteClassifier(LinearSVC_classifier, SGD_classifier, 
                                 BNB_classifier,MNB_classifier, classifier)
print("voted_classifier Accuracy is : ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification : ", voted_classifier.classify(testing_set[0][0]), "Confidence % : ",
      voted_classifier.confidence(testing_set[0][0])*100)
print("Classification : ", voted_classifier.classify(testing_set[1][0]), "Confidence % : ",
      voted_classifier.confidence(testing_set[1][0])*100)
print("Classification : ", voted_classifier.classify(testing_set[2][0]), "Confidence % : ",
      voted_classifier.confidence(testing_set[2][0])*100)
print("Classification : ", voted_classifier.classify(testing_set[3][0]), "Confidence % : ",
      voted_classifier.confidence(testing_set[3][0])*100)
print("Classification : ", voted_classifier.classify(testing_set[4][0]), "Confidence % : ",
      voted_classifier.confidence(testing_set[4][0])*100)

def sentiment(text):
    feats= find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)