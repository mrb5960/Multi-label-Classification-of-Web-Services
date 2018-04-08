from skmultilearn.adapt import MLkNN, BRkNNaClassifier, BRkNNbClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import time

start = time.time()
est = open("estimated parameters.txt", "w")

for n in range(100, 501, 100):
    input_file = 'd2v_and_top20labels_' + str(n) + 'd.csv'
    print '\n' + input_file
    est.write('\n'+ input_file)
    df = pd.read_csv(input_file)
    last_dim = 'dim_' + str(n)

    X = df.loc[:,'dim_1':last_dim]
    y = df.loc[:,'analytics':'video']

    parameters1 = {'k': range(1,10), 's': [0.5, 0.7, 1.0]}
    parameters2 = {'k': range(1, 10)}
    score = 'f1_weighted'

    clf = GridSearchCV(BRkNNaClassifier(), parameters2, scoring=score)
    clf.fit(X.as_matrix(), y.as_matrix())
    print clf.best_params_, clf.best_score_
    est.write('\n' + str(clf.best_params_) + ' ' + str(clf.best_score_))

    clf = GridSearchCV(BRkNNbClassifier(), parameters2, scoring=score)
    clf.fit(X.as_matrix(), y.as_matrix())
    print clf.best_params_, clf.best_score_
    est.write('\n' + str(clf.best_params_) + ' ' + str(clf.best_score_))

    clf = GridSearchCV(MLkNN(), parameters1, scoring=score)
    clf.fit(X.as_matrix(), y.as_matrix())
    print clf.best_params_, clf.best_score_
    est.write('\n' + str(clf.best_params_) + ' ' + str(clf.best_score_))
    est.write('\n')

est.close()
print time.time() - start