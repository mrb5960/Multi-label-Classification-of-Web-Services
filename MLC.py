from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.ensemble import LabelSpacePartitioningClassifier
#from skmultilearn.cluster import IGraphLabelCooccurenceClusterer
from sklearn.metrics import confusion_matrix, hamming_loss, label_ranking_loss, label_ranking_average_precision_score, f1_score, accuracy_score, jaccard_similarity_score, classification_report, precision_recall_fscore_support
from skmultilearn.adapt import BRkNNaClassifier, BRkNNbClassifier, MLkNN
import pandas as pd
import time



start = time.time()

#df = pd.read_csv("tok&lem_final_data_v2_tfidf.csv")
#df = pd.read_csv("tok&lem_final_data_v3_top5labels_tfidf_300f.csv")
#df = pd.read_csv("wordvectors_and_top20labels.csv")
df = pd.read_csv("weighted_d2v_and_top20labels_100d.csv")
print df.shape

#newdf = df.loc[:,'threeD':'zip codes']
# newdf = df.loc[:,'advertising.1':'voice.1']
# print newdf.shape

#featuresdf = df.loc[:,'ability':'zip']
# featuresdf = df.loc[:,'ability':'xml']
# print featuresdf.shape

#>>>>>>>>>>>>>keep in mind top change the column names below after generating a new csv file<<<<<<<<<<<<<<<<<<

X = df.loc[:,'dim_1':'dim_100']

#X = df.loc[:,'ability':'xml']
# 100 features
#X = df.loc[:,'access':'xml']
# 1000 features
#X = df.loc[:,'ability':'zone']
# 300f 5 labels
#X = df.loc[:,'ability':'xml']
# 50 labels
#y = df.loc[:,'l_advertising':'l_voice']
# 5 labels
#y = df.loc[:,'l_data':'l_tools']
# 10 labels
#y = df.loc[:,'l_analytics':'l_tools']
# 20 labels
y = df.loc[:,'l_analytics':'l_video']
#y = df.loc[:,'analytics':'video']

targets = y.columns.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#classifier = BinaryRelevance(GaussianNB())
#classifier = BinaryRelevance(tree.DecisionTreeClassifier())
#classifier = ClassifierChain(tree.DecisionTreeClassifier())
classifier = LabelPowerset(tree.DecisionTreeClassifier())
#classifier = MLkNN(k=5)
#classifier = BRkNNaClassifier(k=5)
#ptclassifier = LabelPowerset(tree.DecisionTreeClassifier())
#clusterer = IGraphLabelCooccurenceClusterer('fastgreedy', weighted=True, include_self_edges=True)
#classifier = LabelSpacePartitioningClassifier(ptclassifier, clusterer)

classifier.fit(X_train.as_matrix(), y_train.as_matrix())

predictions = classifier.predict(X_test.as_matrix())

loss = hamming_loss(y_test.as_matrix(), predictions)
print 'Hamming loss: ', loss

#acc = accuracy_score(y_test.as_matrix(), predictions)
#print 'accuracy: ', acc

#lrloss = label_ranking_loss(y_test, predictions.toarray())
#lrap = label_ranking_average_precision_score(y_test, predictions.toarray())
#print "LRLOSS: best value 0: ", lrloss
#print "LRAP: best value 1: ", lrap

#macro_score = f1_score(y_test, predictions.toarray(), average='macro')
micro_score = f1_score(y_test, predictions.toarray(), average='micro')
weighted_score = f1_score(y_test, predictions.toarray(), average='weighted')
sampled_score = f1_score(y_test, predictions.toarray(), average='samples')
#jac = jaccard_similarity_score(y_test, predictions.toarray())
# equivalent to precision_recall_fscore_support with average='weighted'
#report = classification_report(y_test, predictions.toarray(), target_names=targets)
#prfs = precision_recall_fscore_support(y_test, predictions.toarray())
#cm = confusion_matrix(y_test.values.argmax(axis=1), predictions.argmax(axis=1))

#print 'macro f1 score: ', macro_score
print 'micro f1 score: ', micro_score
print 'weighted f1 score: ', weighted_score
#print 'Jaccard: ', jac
print 'Sample score: ', sampled_score
#print report
#print prfs
#print cm

end = time.time()

print str(end - start)