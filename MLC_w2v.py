from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from skmultilearn.adapt import BRkNNaClassifier, BRkNNbClassifier, MLkNN
from skmultilearn.ensemble import RakelD, RakelO
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import hamming_loss, f1_score, precision_score, recall_score, zero_one_loss
from sklearn.model_selection import cross_val_score, cross_validate
import numpy as np
import warnings
import pandas as pd
import time

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

start = time.time()

out_file = 'output_' + str(start) + '.txt'
results = open(out_file, "w")

for n in range(100, 101, 200):
    #n = 300
    #input_file = 'd2v_and_all_labels_' + str(n) + 'd.csv'
    #input_file = 'd2v_and_all_labels_300d.csv'
    input_file = 'weighted_d2v_and_top20labels_100d.csv'
    results.write('########################################################################################')
    results.write('\n'+ input_file + '\n')
    df = pd.read_csv(input_file)
    last_dim = 'dim_' + str(n)

    df = pd.read_csv(input_file)

    X = df.loc[:,'dim_1':last_dim]

    # 20 labels
    y = df.loc[:,'l_analytics':'l_video']
    #y = df.loc[:,'analytics':'tools']

    targets = y.columns.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # >>>>>>>>>>>>>>>>cross validation remaining <<<<<<<<<<<<<<<<<<

    br_nb = BinaryRelevance(GaussianNB())
    br_dt = BinaryRelevance(DecisionTreeClassifier())
    br_lr = BinaryRelevance(LogisticRegression())
    cc_nb = ClassifierChain(GaussianNB())
    cc_dt = ClassifierChain(DecisionTreeClassifier())
    cc_lr = ClassifierChain(LogisticRegression())
    lp_nb = LabelPowerset(GaussianNB())
    lp_dt = LabelPowerset(DecisionTreeClassifier())
    lp_lr = LabelPowerset(LogisticRegression())
    # rkd_nb = RakelD(GaussianNB(),labelset_size=2)
    # rkd_dt = RakelD(DecisionTreeClassifier(),labelset_size=2)
    # rko_nb = RakelO(GaussianNB(),labelset_size=2)
    # rko_dt = RakelO(DecisionTreeClassifier(),labelset_size=2)

    #classifiers = [br_nb, br_dt, cc_nb, cc_dt, lp_nb, lp_dt, brknn_a, brknn_b, mlknn, rkd_nb, rkd_dt, rko_nb, rko_dt]
    #classifiers = [br_nb, br_dt, cc_nb, cc_dt, lp_nb, lp_dt, brknn_a, brknn_b, mlknn]
    classifiers = {#'Binary Relevance with Gaussian Naive Bayes' : br_nb,
                    #'Binary Relevance with Decision Tree': br_dt,
                   'Binary Relevance with Logistic Regression': br_lr,
                    #'Classifier Chain with Gaussian Naive Bayes': cc_nb,
                    #'Classifier Chain with Decision Tree': cc_dt,
                   'Classifier Chain with Logistic Regression': cc_lr,
                    #'Label Powerset with Gaussian Naive Bayes': lp_nb,
                    #'Label Powerset with Decision Tree': lp_dt,
                   'Label Powerset with Logistic Regression': lp_lr
                   }

    for name, classifier in classifiers.items():
        t0 = time.time()
        print name
        results.write('\n' + name)

        classifier.fit(X_train.as_matrix(), y_train.as_matrix())

        predictions = classifier.predict(X_test.as_matrix())

        # hloss = hamming_loss(y_test.as_matrix(), predictions)
        # print 'Hamming loss: ', round(hloss,2)
        # results.write('\n' + 'Hamming loss: ' + str(round(hloss,2)))
        #
        # z_o_loss = zero_one_loss(y_test.as_matrix(), predictions)
        # print 'Zero one loss: ', round(z_o_loss, 2)
        # results.write('\n' + 'Zero one loss: ' + str(round(z_o_loss, 2)))

        # precision = precision_score(y_test.as_matrix(), predictions)
        # print 'Precision: ', round(precision, 2)
        # results.write('\n' + 'Precision: ' + str(round(precision, 2)))
        #
        # recall  = recall_score(y_test.as_matrix(), predictions)
        # print 'Recall: ', round(recall, 2)
        # results.write('\n' + 'Recall: ' + str(round(recall, 2)))
        #
        # weighted_score = f1_score(y_test, predictions, average='weighted')
        # print 'weighted f1 score: ', round(weighted_score,2)
        # results.write('\n' + 'Weighted f1 score: ' + str(round(weighted_score, 2)))
        #
        # scores = cross_val_score(classifier, X, y, cv=5, scoring='f1_weighted')
        # print 'Cross validated weighted f1 score: '
        # print round(np.mean(scores), 2)
        # results.write('\n' + 'Cross validated weighted f1 score' + str(round(np.mean(scores), 2)))
        # results.write(scores)

        scoring = ['precision_weighted', 'recall_weighted', 'f1_weighted']
        scores = cross_validate(classifier, X.as_matrix(), y.as_matrix(), cv=10, scoring=scoring,
                                return_train_score=False)

        t1 = time.time()
        precision = round(np.mean(scores['test_precision_weighted']), 2)
        recall = round(np.mean(scores['test_recall_weighted']), 2)
        f1 = round(np.mean(scores['test_f1_weighted']), 2)
        time_taken = t1 - t0

        results.write('\n' + 'Precision: ' + str(precision))
        results.write('\n' + 'Recall: ' + str(recall))
        results.write('\n' + 'F1 score: ' + str(f1))
        results.write('\n' + 'Time taken: ' + str(time_taken))
        print 'Precision: ', precision
        print 'Recall: ', recall
        print 'F1: ', f1
        print 'Time taken: ', time_taken

        print '\n'
        results.write('\n')

results.close()
end = time.time()

print str(end - start)