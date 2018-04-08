from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from skmultilearn.adapt import BRkNNaClassifier, BRkNNbClassifier, MLkNN
from skmultilearn.ensemble import RakelD, RakelO
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import hamming_loss, f1_score, zero_one_loss, precision_score, recall_score
from sklearn.model_selection import cross_val_score, cross_validate
import numpy as np
import warnings
import pandas as pd
import time

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

start = time.time()

input_file = "tok&lem&postagged_final_data_v4_top20labels_tfidf_100f.csv"

df = pd.read_csv(input_file)

X = df.loc[:,'access':'xml']
y = df.loc[:,'l_analytics':'l_video']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

for k in range(4, 5):
    t0 = time.time()
    print 'k =', k
    classifier = BRkNNbClassifier(k=k)

    # classifier.fit(X_train.as_matrix(), y_train.as_matrix())
    # predictions = classifier.predict(X_test.as_matrix())
    #
    # hloss = hamming_loss(y_test.as_matrix(), predictions)
    # print 'Hamming loss: ', round(hloss, 2)
    #
    # z_o_loss = zero_one_loss(y_test.as_matrix(), predictions)
    # print 'Zero one loss: ', round(z_o_loss, 2)
    #
    # precision = precision_score(y_test.as_matrix(), predictions, average='weighted')
    # print 'Precision: ', round(precision, 2)
    # #print 'Precision: ', round(np.mean(precision), 2)
    #
    # recall = recall_score(y_test.as_matrix(), predictions, average='weighted')
    # print 'Recall: ', round(recall, 2)
    # #print 'Recall: ', round(np.mean(recall), 2)
    #
    # f1 = f1_score(y_test, predictions, average='weighted')
    # print 'weighted f1 score: ', round(f1, 2)
    #
    #
    # scores = cross_val_score(classifier, X.as_matrix(), y.as_matrix(), cv=5, scoring='f1_weighted')
    # print 'Cross validated weighted f1 score: ', round(np.mean(scores), 2)
    #
    # pw = cross_val_score(classifier, X.as_matrix(), y.as_matrix(), cv=5, scoring='precision_weighted')
    # print 'Cross validated weighted precision: ', round(np.mean(pw), 2)
    #
    # rw = cross_val_score(classifier, X.as_matrix(), y.as_matrix(), cv=5, scoring='recall_weighted')
    # print 'Cross validated weighted recall: ', round(np.mean(rw), 2)

    scoring = ['precision_weighted', 'recall_weighted', 'f1_weighted']
    scores = cross_validate(classifier, X.as_matrix(), y.as_matrix(), cv=10, scoring=scoring, return_train_score=False)
    precision = round(np.mean(scores['test_precision_weighted']), 2)
    recall = round(np.mean(scores['test_recall_weighted']), 2)
    f1 = round(np.mean(scores['test_f1_weighted']), 2)
    t1 = time.time()
    time_taken = t1 - t0
    print 'Precision: ', precision
    print 'Recall: ', recall
    print 'F1: ', f1
    print 'Time taken: ', time_taken

end = time.time()

print str(end - start)