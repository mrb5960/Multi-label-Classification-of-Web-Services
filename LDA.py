import nltk
import os
import string
import re
from math import log
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import pickle
from operator import itemgetter

# some more words that need to be ignored
add = ['service', 'api', 'provides', '', 'allows', 'support', 'methods', 'services', 'users', 'include', 'search',
       'also', 'data', 'information', 'access', 'content', 'xml', 'json', 'web', 'site']

stop_words = set(stopwords.words("english"))

# adding the above list of words into stopwords
stop_words = stop_words.union(add)

# using snowball stemmer
stemmer = SnowballStemmer('english')

# function that performs stemming on a list of words
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

# function that removes the stopwords
def remove_stopwords(str):
    str_list = nltk.word_tokenize(str)
    cleaned_text = list(filter(lambda x: x not in stop_words, str_list))
    return cleaned_text

def get_tokens():
    '''
    function that creates tokens out of strings
    performs preprocessing
    Writes the generated tokens to a file for later use
    :return:
    '''
    path = "C:\\Study\\RIT\\Web Services\\Assignments\\A4\\GeneratedFiles\\"
    out_path = "C:\\Study\\RIT\\Web Services\\Assignments\\A4\\GeneratedTokens\\"

    for subdir, dirs, files in os.walk(path):
        # for every file in the folder
        for file in files:
            file_path = subdir + file
            # open the file
            service = open(file_path, 'r')
            # read the text
            text = service.read()
            # convert to lower case
            lower = text.lower()
            # remove special symbols
            remove_special_symbols = re.sub('[^a-zA-Z ]', ' ', lower)
            # remove extra spaces
            remove_spaces = re.sub("\s\s+", " ", remove_special_symbols)
            # remove stopwords
            remove_xml = remove_stopwords(remove_spaces)
            # write contents to a file
            out_file_path = out_path + file.title().split(".")[0] + '.txt'
            out_file = open(out_file_path, "w+")

            out_str = ""
            for id in range(0,len(remove_xml)):
                if len(remove_xml[id]) > 2:
                    out_str += remove_xml[id] + " "

            out_file.write(out_str)
            out_file.close()
            #print(out_str)

get_tokens()

path = "C:\\Study\\RIT\\Web Services\\Assignments\\A4\\GeneratedTokens\\"
#out_path = "C:\\Study\\RIT\\Web Services\\Assignments\\A4\\corpus.txt"
corpus = []

for subdir, dirs, files in os.walk(path):
        for file in files:
            # for each file in the directory
            file_path = subdir + file
            service = open(file_path, 'r')
            text = service.read()
            # remove stop words
            rem_stop = remove_stopwords(text)
            # perform stemming
            stemmed = stem_tokens(rem_stop, stemmer)
            # add contents to a 2D list
            corpus.append(stemmed)

# pickling the object for later use
pickle_out = open("stemmed_corpus","wb")
pickle.dump(corpus, pickle_out)
pickle_out.close()

# list that contains the names of files or web services
names = []
for subdir, dirs, files in os.walk(path):
    for file in files:
        names.append(file.title())

# get the pickled 2D list
pickle_in = open("stemmed_corpus","rb")
stemmed_corpus = pickle.load(pickle_in)
#print(stemmed_corpus[0])

# converting the corpus to a dictionary
dictionary = corpora.Dictionary(stemmed_corpus)

# print the tokens and their ids
print(dictionary.token2id)

# generate bag of words
bag_of_words = [dictionary.doc2bow(doc) for doc in stemmed_corpus]
#print(bag_of_words[0])

# build a topic model using LDA
ldamodel = gensim.models.LdaModel(bag_of_words, num_topics=20, id2word=dictionary, passes=10)

#pickle the object for later use
pickle_out = open("ldamodel1","wb")
pickle.dump(ldamodel, pickle_out)
pickle_out.close()

#ldamodel.get_document_topics()

# get the pickled model
pickle_in = open("ldamodel1","rb")
ldamodel = pickle.load(pickle_in)

# print generated topics
for i in range(0,20):
    print(ldamodel.get_topic_terms(i))
    print(i, ')', ldamodel.print_topic(i, 10))

# print the first element of the stemmed corpus
print(stemmed_corpus[0])

# print topic distributions for first bag of words
print(ldamodel[bag_of_words[0]])
print(ldamodel.get_document_topics(bag_of_words[0]))

# print bag of words sample
print(bag_of_words[0])

# dictionary of topics
topics = dict()
#initialize dictionary
for i in range(0, 20):
    temp = []
    topics[i] = temp

#adding web services to the topics
for i in range(0, len(bag_of_words)):
    # get the topics for each bag of words of web services
    doc_top = ldamodel.get_document_topics(bag_of_words[i])

    # sort the topics based on their probability
    doc_top_sorted = sorted(doc_top,key=itemgetter(1))

    # providing a threshold to avoid noise
    if doc_top_sorted[0][1] > 0.4:
        # getting the first topic
        top_topic = doc_top_sorted[0][0]

        # adding the web service to the list of selected topic
        topics[top_topic].append(names[i])

# displaying the list of web services for topic id 11
print(topics[11])

