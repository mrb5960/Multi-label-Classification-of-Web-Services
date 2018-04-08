from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectKBest
import pandas as pd

df = pd.read_csv("tok&lem_final_data_v3_top10labels.csv")
vectorizer = CountVectorizer(lowercase=True,stop_words='english')

data = df['api_desc']
target = df.loc[:,'analytics':'tools']

X = vectorizer.fit_transform(data)
print X.shape

print vectorizer.get_feature_names()
# X_new = SelectKBest(chi2, k=4).fit_transform(X, target)
# print X_new

