from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from skmultilearn.adapt import BRkNNaClassifier, BRkNNbClassifier, MLkNN
from skmultilearn.ensemble import RakelD, RakelO
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import hamming_loss, f1_score, precision_score, recall_score, zero_one_loss
from sklearn.model_selection import cross_val_score, cross_validate
import numpy as np
import warnings
import pandas as pd
import time

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

start = time.time()

#out_file = 'tfidf_output_' + str(start) + '.txt'
#results = open(out_file, "w")

input_file = 'tok&lem&postagged_final_data_v4_top20labels_tfidf_1000f.csv'

#results.write('########################################################################################')
#results.write('\n'+ input_file + '\n')
df = pd.read_csv(input_file)

X = df.loc[:,'access':'xml']
y = df.loc[:,'l_analytics':'l_video']

targets = y.columns.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# >>>>>>>>>>>>>>>>cross validation remaining <<<<<<<<<<<<<<<<<<

br_nb = BinaryRelevance(GaussianNB())
br_dt = BinaryRelevance(DecisionTreeClassifier())
br_lr = BinaryRelevance(LogisticRegression())
br_mlpc = BinaryRelevance(MLPClassifier())
br_mlpr = BinaryRelevance(MLPRegressor())
cc_nb = ClassifierChain(GaussianNB())
cc_dt = ClassifierChain(DecisionTreeClassifier())
cc_lr = ClassifierChain(LogisticRegression())
cc_mlpc = ClassifierChain(MLPClassifier())
cc_mlpr = ClassifierChain(MLPRegressor())
lp_nb = LabelPowerset(GaussianNB())
lp_dt = LabelPowerset(DecisionTreeClassifier())
lp_lr = LabelPowerset(LogisticRegression())
lp_mlpc = LabelPowerset(MLPClassifier())
lp_mlpr = LabelPowerset(MLPRegressor())


classifiers = {#'Binary Relevance with Gaussian Naive Bayes' : br_nb,
#                'Binary Relevance with Decision Tree': br_dt,
                #'Binary Relevance with Logistic Regression': br_lr,
                'Binary Relevance with MLP Classifier': br_mlpc,
                #'Binary Relevance with MLP Regressor': br_mlpr,
#                'Classifier Chain with Gaussian Naive Bayes': cc_nb,
#                'Classifier Chain with Decision Tree': cc_dt,
                #'Classifier Chain with Logistic Regression': cc_lr,
                #'Classifier Chain with MLP Classifier': cc_mlpc,
                #'Classifier Chain with MLP Regressor': cc_mlpr,
#                'Label Powerset with Gaussian Naive Bayes': lp_nb,
#                'Label Powerset with Decision Tree': lp_dt,
                #'Label Powerset with Logistic Regression': lp_lr
                #'Label Powerset with MLP Classifier': lp_mlpc,
                #'Label Powerset with MLP Regressor': lp_mlpr
               }

for name, classifier in classifiers.items():
    t0 = time.time()
    print name
    #results.write('\n' + name)

    classifier.fit(X_train.as_matrix(), y_train.as_matrix())

    predictions = classifier.predict(X_test.as_matrix())

    scoring = ['precision_weighted', 'recall_weighted', 'f1_weighted']
    scores = cross_validate(classifier, X.as_matrix(), y.as_matrix(), cv=10, scoring=scoring,
                            return_train_score=False)

    t1 = time.time()
    precision = round(np.mean(scores['test_precision_weighted']), 2)
    recall = round(np.mean(scores['test_recall_weighted']), 2)
    f1 = round(np.mean(scores['test_f1_weighted']), 2)
    time_taken = t1 - t0

    # results.write('\n' + 'Precision: ' + str(precision))
    # results.write('\n' + 'Recall: ' + str(recall))
    # results.write('\n' + 'F1 score: ' + str(f1))
    # results.write('\n' + 'Time taken: ' + str(time_taken))
    print 'Precision: ', precision
    print 'Recall: ', recall
    print 'F1: ', f1
    print 'Time taken: ', time_taken

    print '\n'
    #results.write('\n')

#results.close()
end = time.time()

print str(end - start)