import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from nltk.stem import WordNetLemmatizer, PorterStemmer
import operator
import time
from gensim.models import Word2Vec
from textblob import TextBlob
from nltk import pos_tag
import numpy as np
from collections import OrderedDict
from gensim.models import KeyedVectors
import math

def cleanData():
    df = pd.read_csv("final_data.csv")

    newdf = pd.DataFrame(columns=['A'])

    count = 0
    for index, row in df.iterrows():
        if row['none'] != 1:
            newdf = newdf.append(row)
            count = count + 1
            print(count)

    newdf = newdf.drop(['A'], axis=1)

    print(len(df), '', len(newdf))

    newdf.to_csv('final_data_v2.csv')

def createTestFile():
    df = pd.read_csv('final_data_v2.csv')

    newdf = df.head(n=10)

    newdf.to_csv("test_file.csv", index=False)

def tokenize():

    print str(datetime.now())

    #df = pd.read_csv('test_file.csv')
    df = pd.read_csv('final_data_v2.csv')

    stop_words = set(stopwords.words('english'))
    other_words = ('also', 'nthe')
    stop_words.add('also')
    stop_words.add('nthe')

    lem = WordNetLemmatizer()
    stemmer = PorterStemmer()

    rows_to_delete = []

    for index, row in df.iterrows():
        #print 'Converting to lowercase...'
        desc = row['api_desc'].lower()

        #print 'removing special symbols...'
        remove_special_symbols = re.sub('[^a-zA-Z ]', ' ', desc)

        #print 'removing spaces...'
        remove_spaces = re.sub("\s\s+", " ", remove_special_symbols)

        #print 'creating tokens...'
        tokens = word_tokenize(remove_spaces)

        # lemmatize the tokens
        lemmetized_tokens = [lem.lemmatize(w) for w in tokens]

        # stem the tokens
        #stemmed_tokens = [stemmer.stem(w) for w in tokens]

        #print 'removing stopwords...'
        remove_stop_words = [w for w in lemmetized_tokens if not w in stop_words]

        #print 'removing short words...'
        remove_short_words = [w for w in remove_stop_words if len(w) > 2]

        # keep nouns only
        #tagged = pos_tag(remove_short_words)
        #keep_nouns = [word for word,pos in tagged if (pos == 'NN' or pos == 'NNS' or pos == 'NNP' or pos == 'NNPS')]

        token_string = ""

        #print 'creating token string...'
        words = 0
        for w in remove_short_words:
        #for w in keep_nouns:
            words += 1
            token_string += w + " "

        # remove rows whose description is less than 50 words
        if words < 15:
            rows_to_delete.append(index)

        token_string = token_string.strip()

        df.at[index, 'api_desc'] = token_string

    #print len(rows_to_delete)
    df = df.drop(df.index[rows_to_delete])
    #print df.shape
    #df = df.drop(df[df.api_desc == ""].index)
    print 'Writing to csv file...'
    df.to_csv("tok&lem_final_data_v4.csv", index=False)

    print str(datetime.now())

def getTFIDF():

    print str(datetime.now())

    #df = pd.read_csv("tokenized test file.csv")
    #df = pd.read_csv("tok&lem_final_data_v2.csv")
    #df = pd.read_csv("tok&lem_final_data_v3_top20labels.csv")
    df = pd.read_csv("tok&lem&postagged_final_data_v4_top20labels.csv")
    print df.shape

    newdf = pd.DataFrame(columns=['A'])

    # create a dictionary which can be used for calculating TF-IDF
    token_dict = OrderedDict()
    api_names = []
    #count = 0
    #duplicates = []

    print 'creating dictionary...'
    for index, row in df.iterrows():

        api_name = row['api_name']
        api_names.append(api_name)
        api_desc = row['api_desc']
        # if api_name in token_dict:
        #     count += 1
        #     duplicates.append(api_name)
        token_dict[api_name] = api_desc

    # print('Null values in final', df.isnull().any().any())
    # nan_rows = df[df.isnull().any(1)]
    # print(nan_rows)
    #print len(token_dict)
    #print sorted(api_names)
    #print duplicates

    print 'generating TF-IDF matrix...'
    tfidf = TfidfVectorizer(max_features=1000, max_df=0.95)
    tfs = tfidf.fit_transform(token_dict.values())
    feature_names = tfidf.get_feature_names()
    #print(feature_names)
    #print 'Number of features: ', len(feature_names)
    #print tfs.shape
    dense = tfs.todense()
    #print dense.shape

    tfsdf = pd.DataFrame(tfs.toarray())
    tfsdf = tfsdf.round(4)
    tfsdf.columns = feature_names

    newdf['api_name'] = api_names

    print 'concatenating dataframes...'
    newdf = pd.concat([newdf, tfsdf], axis=1)

    # 50 labels
    #targets = df.loc[:, 'advertising':'voice']
    # 5 labels
    #targets = df.loc[:, 'data':'tools']
    # 10 labels
    #targets = df.loc[:, 'analytics':'tools']
    # 20 labels
    targets = df.loc[:, 'analytics':'video']
    # all labels
    #targets = df.loc[:, '3d':'zip codes']
    targets = targets.rename(columns=lambda x: 'l_' + x)
    #newdf = pd.concat([newdf, df.loc[:,'3d':'zip codes']], axis=1)
    newdf = pd.concat([newdf, targets], axis=1)

    newdf = newdf.drop(['A'], axis=1)
    print newdf.columns

    print 'writing to csv file...'
    #newdf.to_csv("tok&lem_final_data_v2_tfidf.csv", index=False)
    #newdf.to_csv("tok&lem_final_data_v2_tfidf_500f.csv", index=False)
    #newdf.to_csv("tok&lem_final_data_v3_top5labels_tfidf_300f.csv", index=False)
    #newdf.to_csv("tok&lem_final_data_v3_top20labels_tfidf_100f.csv", index=False)
    newdf.to_csv("tok&lem&postagged_final_data_v4_top20labels_tfidf_1000f.csv", index=False)

    print str(datetime.now())

def mergeLabels():
    print str(datetime.now())

    df = pd.read_csv("tok&lem_final_data_v2_tfidf.csv")
    print df.shape

    newdf = df.loc[:,'3d':'zip codes']
    print newdf.shape

    for index,rows in newdf.iterrows():
        labelstr = ""
        for col in rows:
            labelstr += str(col)
        df.at[index, 'Merged labels'] = labelstr

    df.to_csv("merged_labels.csv", index=False)

    print str(datetime.now())

def checkLabelFreq():
    print str(datetime.now())

    df = pd.read_csv("merged_labels.csv")

    label_dict = {}

    # for index, row in df.iterrows():
    #     if row['Merged labels'] in label_dict.keys():
    #         label_dict[row['Merged labels']] += 1
    #     else:
    #         label_dict[row['Merged labels']] = 1

    for index, row in df.iterrows():
        if row['Merged labels'] in label_dict.keys():
            label_dict[row['Merged labels']].append(row['api_name'])
        else:
            api_list = []
            api_list.append(row['api_name'])
            label_dict[row['Merged labels']] = api_list

    #desc = sorted(label_dict.items(), key=operator.itemgetter(1), reverse=True)

    newdf = pd.DataFrame(columns=['label', 'count', 'web services'])

    #print label_dict.iteritems().next()

    index = 0
    # for desc in label_dict:
    #     newdf.at[index, 'label'] = l[0]
    #     newdf.at[index, 'count'] = len(l[1])
    #     newdf.at[index, 'web services'] = l[1]
    #     index += 1

    for key, value in label_dict.items():
        newdf.at[index, 'label'] = key
        newdf.at[index, 'count'] = len(value)
        newdf.at[index, 'web services'] = value
        index += 1

    newdf.to_csv('unique labels and web services.csv', index=False)

    print str(datetime.now())

def getTopLabels(n):
    '''
    Function to get the top n labels
    :param n: top 'n' labels
    :return: list of top labels
    '''
    print str(datetime.now())

    #df = pd.read_csv("tok&stem_final_data_v3.csv")
    df = pd.read_csv("tok&lem&postagged_final_data_v4.csv")

    newdf = df.loc[:,'3d':'zip codes']

    cardinality = {}
    for column in newdf:
        cardinality[column] = df[column].sum()

    desc = sorted(cardinality.items(), key=operator.itemgetter(1), reverse=True)

    # for l in desc:
    #     print l[0], " ", l[1]
    #
    # label_sum = 0
    #
    # for l in desc:
    #     if l[1] > 500:
    #         label_sum += l[1]
    #
    # print label_sum

    topLabels = []
    for i in range(0,n):
        topLabels.append(desc[i][0])

    #print len(topLabels)

    print str(datetime.now())

    return topLabels

def getTopLabelsWithLC(n):
    print str(datetime.now())

    df = pd.read_csv("tok&stem_final_data_v3.csv")

    newdf = df.loc[:, '3d':'zip codes']

    cardinality = {}
    for column in newdf:
        cardinality[column] = df[column].sum()

    desc = sorted(cardinality.items(), key=operator.itemgetter(1), reverse=True)

    topLabels = []
    for i in range(0, n):
        topLabels.append(desc[i][0])

    topLabels.append('api_name')
    topLabels.append('api_desc')

    cols_to_delete = []
    for cols in df.columns.values:
        if cols not in topLabels:
            cols_to_delete.append(cols)

    df = df.drop(cols_to_delete, axis=1)
    print 'Shape after deleting columns: ', df.shape

    print 'Top columns', df.columns

    rows_to_delete = []

    newdf = df.ix[:, 2:]

    #print newdf
    for index, rows in newdf.iterrows():
        if rows.sum() == 0:
            rows_to_delete.append(index)

    print 'Number of rows to be deleted: ', len(rows_to_delete)

    # remove the following if you want to include services with zero categories
    newdf = newdf.drop(newdf.index[rows_to_delete])
    print 'After deleting rows: ', newdf.shape

    row_count = len(newdf)

    label_count = 0
    for index, rows in newdf.iterrows():
        label_count += rows.sum()

    labelCard = label_count / float(row_count)

    print 'Label count: ', label_count
    print 'Row count: ', row_count
    print 'Label cardinality: ', labelCard

def keepTopLabels():
    '''
    Function to keep top n labels along with tokenized and lemmatized data
    :return: None
    '''
    print str(datetime.now())

    #df = pd.read_csv("tok&lem_final_data_v3.csv")
    df = pd.read_csv("tok&lem_final_data_v4.csv")

    print df.shape

    topLabels = getTopLabels(20)
    topLabels.append('api_name')
    topLabels.append('api_desc')

    print topLabels

    cols_to_delete = []
    for cols in df.columns.values:
        if cols not in topLabels:
            cols_to_delete.append(cols)

    df = df.drop(cols_to_delete, axis=1)
    print df.shape

    print df.columns

    rows_to_delete = []

    # for 20 labels
    newdf = df.loc[:,'analytics':'video']
    # for 10 labels
    #newdf = df.loc[:, 'analytics':'tools']

    # for 5 labels
    #newdf = df.loc[:, 'data':'tools']

    for index, rows in newdf.iterrows():
        if rows.sum() == 0:
            rows_to_delete.append(index)

    print len(rows_to_delete)
    # remove the following if you want to include services with zero categories

    df = df.drop(df.index[rows_to_delete])
    print df.shape

    #df.to_csv("tok&lem_final_data_v3_top5labels.csv", index=False)
    df.to_csv("tok&lem_final_data_v4_top20labels.csv", index=False)

    print str(datetime.now())

def keepTopLabels2():
    '''
    Function to keep top labels and document vectors
    :return: None
    '''
    for n in range(100, 501, 100):
        name = 'd2v_and_all_lables_' + str(n) + 'd.csv'
        df = pd.read_csv(name)
        #print df.columns.values

        topLabels = getTopLabels(20)

        targets = df.loc[:,'3d':'zip codes']
        cols_to_delete = []
        for col in targets.columns.values:
            if col not in topLabels:
                cols_to_delete.append(col)

        #print cols_to_delete
        df = df.drop(cols_to_delete, axis=1)

        print df.shape

        rows_to_delete = []
        # for 10 labels
        #newdf = df.loc[:, 'analytics':'tools']
        # for 20 labels
        newdf = df.loc[:, 'analytics':'video']

        # removes services that do not belong to any category
        for index, rows in newdf.iterrows():
            if rows.sum() == 0:
                rows_to_delete.append(index)

        print len(rows_to_delete)
        # remove the following if you want to include services with zero categories

        df = df.drop(df.index[rows_to_delete])
        print df.shape

        opfile = 'd2v_and_top20lables_' + str(n) + 'd.csv'
        df.to_csv(opfile, index=False)


def getLabelCardinality():
    df = pd.read_csv("tok&lem_final_data_v3_top20labels.csv")
    #df = pd.read_csv("tok&lem&postagged_final_data_v4.csv")

    #newdf = df.loc[:,'advertising':'voice']
    newdf = df.loc[:, 'analytics':'tools']
    #newdf = df.loc[:, '3d':'zip codes']

    row_count = len(newdf)

    label_count = 0
    for index,rows in newdf.iterrows():
        label_count += rows.sum()

    labelCard = label_count/float(row_count)


    print label_count
    print row_count
    print labelCard

def renameDuplicateCols(df):
    #df = pd.read_csv("tok&lem_final_data_v3_top50labels.csv")

    df = df.rename(columns=lambda x: 'l_' + x if '.1' in x else x)
    df = df.rename(columns=lambda x: x.replace('.1','') if '.1' in x else x)

    return df

def getWordVectors():

    start = time.time()

    # >>>>>>>>>>>>>>>> Creating corpus for word2vec model >>>>>>>>>>>>>>>>>>>>
    #
    # df = pd.read_csv('final_data_v2.csv')
    #
    # rows_to_delete = []
    #
    # for index, row in df.iterrows():
    #     # print 'Converting to lowercase...'
    #     desc = row['api_desc'].lower()
    #
    #     # print 'removing special symbols...'
    #     remove_special_symbols = re.sub('[^a-zA-Z ]', ' ', desc)
    #
    #     # print 'removing spaces...'
    #     remove_spaces = re.sub("\s\s+", " ", remove_special_symbols)
    #
    #     # remove single characters
    #     remove_single_chars = re.sub("\s[a-zA-Z]\s", "   ", remove_spaces)
    #
    #     remove_spaces = re.sub("\s\s+", " ", remove_single_chars)
    #
    #     if len(remove_spaces.split(" ")) < 20:
    #         rows_to_delete.append(index)
    #
    #     df.at[index,'api_desc'] = remove_spaces
    #
    # df = df.drop(df.index[rows_to_delete])
    #
    # print len(rows_to_delete)
    #df.to_csv("final_data_w2v.csv", index=False)
    #
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    #>>>>>>>>>> Building word2vec models >>>>>>>>>>>>>>
    #
    # df = pd.read_csv("final_data_w2v.csv")
    # descriptions = []
    #
    # for index, row in df.iterrows():
    #     desc = row['api_desc']
    #     sentences = desc.split(".")
    #
    #     for s in sentences:
    #         tokens = []
    #         split1 = s.split(" ")
    #         for t in split1:
    #             if len(t) > 1:
    #                 tokens.append(t)
    #         descriptions.append(tokens)
    #
    # #print len(descriptions)
    #
    # print 'Building model'
    #
    # for n in range(100, 501, 100):
    #     model = Word2Vec(descriptions, size=n)
    #     name = 'word2vec_' + str(n) + 'd.bin'
    #     model.save(name)
    #
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>> Building document vectors >>>>>>>>>>>>>>>>

    df = pd.read_csv("final_data_w2v.csv")
    print df.shape

    for n in range(100, 501, 100):
        name = 'word2vec_' + str(n) + 'd.bin'
        model = Word2Vec.load(name)

        newdf = pd.DataFrame(columns=['A'])

        x = []
        api_names = []
        for index, row in df.iterrows():
            api_names.append(row['api_name'])
            doc = word_tokenize(row['api_desc'])
            # if len(doc) == 0:
            #     print index

            doc = [w for w in doc if w in model.wv.vocab]
            # if len(doc) == 0:
            #     continue
            d2v = np.mean(model.wv[doc], axis=0)
            x.append(d2v)

        X = np.array(x)
        cols = []
        for i in range(1, n+1):
            name = 'dim_' + str(i)
            cols.append(name)

        vecdf = pd.DataFrame(X, columns=cols)

        newdf['api_name'] = api_names
        newdf = pd.concat([newdf, vecdf], axis=1)

        targets = df.loc[:,'3d':'zip codes']
        newdf = pd.concat([newdf, targets], axis=1)
        newdf = newdf.drop(['A'], axis=1)
        print newdf.shape

        opfile = 'd2v_and_all_labels_' + str(n) + 'd.csv'
        newdf.to_csv(opfile, index=False)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    print time.time() - start

def getDocVectors():
    start = time.time()
    # make api names as indices
    # df = pd.read_csv("final_data_w2v.csv")
    df = pd.read_csv("tok&lem_final_data_v4_top20labels.csv")
    # df.set_index('api_name')
    print df.shape

    tfdf = pd.read_csv("tok&lem_final_data_v4_top20labels_tfidf_1000f.csv")
    print tfdf.shape
    # df.set_index('api_name')
    #tf_columns = tfdf.columns.values

    name = 'word2vec_300d.bin'
    model = Word2Vec.load(name)

    #model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    #model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')

    newdf = pd.DataFrame(columns=['A'])

    x = []
    api_names = []
    # wordvectors = []
    for index, row in df.iterrows():
        api_names.append(row['api_name'])
        # api_names.append(str(index))
        doc = word_tokenize(row['api_desc'])
        # if len(doc) == 0:
        #     print index

        doc = [w for w in doc if w in model.wv.vocab]

        #doc = [w for w in doc if w in tf_columns]

        if len(doc) == 0:
            print index
        weightedWordVectors = []

        for word in doc:
            api_name = index
            # try:
            # weight = tfdf.loc[api_name, word]
            #weight = tfdf.at[index, word]
            # except KeyError:
            #     weight = 0
            #     continue
            wordvector = model.wv[word]
            wwv = wordvector
            #wwv = wordvector * weight

            # if wwv.size == 0:
            #     print index

            # print wordvector, ' ', weight, ' ', wwv
            # print api_name, ' ', word, ' ', weight, ' ', wwv
            weightedWordVectors.append(wwv)

        d2v = np.mean(weightedWordVectors, axis=0)
        # if math.isnan(d2v):
        #     print index
        # if d2v.size == 0:
        #     print index
        # print d2v.shape
        # print d2v
        x.append(d2v)

    X = np.array(x)
    # print x.shape
    print X.shape
    cols = []
    for i in range(1, 301):
        name = 'dim_' + str(i)
        cols.append(name)

    df.reset_index()
    print df.columns.values
    vecdf = pd.DataFrame(X, columns=cols)
    vecdf = vecdf.round(8)

    newdf['api_name'] = df['api_name']
    newdf = pd.concat([newdf, vecdf], axis=1)

    # 20 labels
    targets = tfdf.loc[:, 'l_analytics':'l_video']
    newdf = pd.concat([newdf, targets], axis=1)
    newdf = newdf.drop(['A'], axis=1)
    print newdf.shape

    opfile = 'd2v_and_top20labels_300d.csv'
    newdf.to_csv(opfile, index=False)

    print time.time() - start


def getWeightedDocVectors():
    start = time.time()
    # make api names as indices
    #df = pd.read_csv("final_data_w2v.csv")
    df = pd.read_csv("tok&lem_final_data_v4_top20labels.csv")
    #df.set_index('api_name')
    print df.shape

    tfdf = pd.read_csv("tok&lem_final_data_v4_top20labels_tfidf_1000f.csv")
    print tfdf.shape
    #df.set_index('api_name')
    tf_columns = tfdf.columns.values

    name = 'word2vec_300d.bin'
    model = Word2Vec.load(name)

    #model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    #model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')

    newdf = pd.DataFrame(columns=['A'])

    x = []
    api_names = []
    #wordvectors = []
    for index, row in df.iterrows():
        api_names.append(row['api_name'])
        #api_names.append(str(index))
        doc = word_tokenize(row['api_desc'])
        # if len(doc) == 0:
        #     print index

        doc = [w for w in doc if w in model.wv.vocab]

        doc = [w for w in doc if w in tf_columns]
        if len(doc) == 0:
            print index
        weightedWordVectors = []
        for word in doc:
            api_name = index
            #try:
                #weight = tfdf.loc[api_name, word]
            weight = tfdf.at[index, word]
            # except KeyError:
            #     weight = 0
            #     continue
            wordvector = model.wv[word]
            wwv = wordvector * weight

            # if wwv.size == 0:
            #     print index

            #print wordvector, ' ', weight, ' ', wwv
            #print api_name, ' ', word, ' ', weight, ' ', wwv
            weightedWordVectors.append(wwv)

        d2v = np.mean(weightedWordVectors, axis=0)
        # if math.isnan(d2v):
        #     print index
        # if d2v.size == 0:
        #     print index
        #print d2v.shape
        #print d2v
        x.append(d2v)

    X = np.array(x)
    #print x.shape
    print X.shape
    cols = []
    for i in range(1, 301):
        name = 'dim_' + str(i)
        cols.append(name)

    df.reset_index()
    print df.columns.values
    vecdf = pd.DataFrame(X, columns=cols)
    vecdf = vecdf.round(8)

    newdf['api_name'] = df['api_name']
    newdf = pd.concat([newdf, vecdf], axis=1)

    #20 labels
    targets = tfdf.loc[:, 'l_analytics':'l_video']
    newdf = pd.concat([newdf, targets], axis=1)
    newdf = newdf.drop(['A'], axis=1)
    print newdf.shape

    opfile = 'weighted_d2v_and_top20labels_300d.csv'
    newdf.to_csv(opfile, index=False)

    print time.time() - start

#createTestFile()
#tokenize()
#getTopLabels(477)
#getTopLabelsWithLC(20)
#keepTopLabels()
#keepTopLabels2()
#getTFIDF()
#mergeLabels()
#checkLabelFreq()
#getLabelCardinality()
#renameDuplicateCols()
#getWordVectors()
#getDocVectors()
getWeightedDocVectors()

'''
for word vectors
1. create n dimensional word2vec model
2. get document vectors
3. keep top labels
'''



#16641 web services, 477 labels, 1000 features
# names of tfidf words and labels were same leading to wrong slicing
