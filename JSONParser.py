import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import re

def JSONToCSV():
    '''
    pickle_file = "Pickled dataframe"
    df = pd.read_csv("api_data.csv")
    print('csv file read')

    newdf = pd.DataFrame(columns=['A'])

    # remove the apis that are no longer available
    for index, row in df.iterrows():
        if row['api_desc'].find('no longer available') == -1:
            newdf = newdf.append(row)

    newdf = newdf.drop(['A'], axis=1)
    print('newdf created')

    # list that stores the labels for every instance
    labelsets = []

    for index, row in newdf.iterrows():
        row['api_tags'] = row['api_tags'].lower()
        labelsets.append(row['api_tags'].split(','))

    #regex = re.compile('[]')
    # for l in labelsets:
    #     for item in l:
    #         re.sub(r'\W+', '', item)


    #print(labelsets)
    print('labelsets created')

    # convert the labelsets to binary labels
    mlb = MultiLabelBinarizer()
    arr = mlb.fit_transform(labelsets)
    #print(len(mlb.classes_), len(arr))
    #print(mlb.classes_, arr)
    print('binarizer complete')

    label_df = pd.DataFrame(arr, columns=mlb.classes_)
    #print('Length of label_df', len(label_df))
    #print(label_df)

    # index has to be reset as removing elements modifies the indexes
    newdf = newdf.reset_index()
    #print('Length of newdf', len(newdf))
    #print(newdf)

    #print('----------------------------------------------------------------')

    #print('Null values in label_df', label_df.isnull().any().any())

    #label_df = label_df.astype(int)

    # concat dataframe containing names and desc with binary labels
    print('concatinating desc, names and labels')
    final = pd.concat([newdf, label_df], axis=1, join_axes=[newdf.index])

    # make api_name as the first column
    cols = ['api_name'] + [col for col in final if col != 'api_name']

    final = final[cols]

    # remove tags are we have binary variables
    final = final.drop(['api_tags'], axis=1)

    # removing newline character and extra spaces form api_desc
    final = final.replace('\n\s+', '', regex=True)

    #print('Null values in final', final.isnull().any().any())

    fileObject = open(pickle_file, 'wb')
    pickle.dump(final, fileObject)
    fileObject.close()

'''
pickle_file = "Pickled dataframe"
fileObject = open(pickle_file, 'r')
final = pickle.load(fileObject)

print(final.columns.values)

    #final = final.set_index('api_name')
    #print(final.index)

    #nan_rows = final[final.isnull().any(1)]
    #print(nan_rows)

    #>>>>>>>>>>>>>>>> make changes after getting all the data <<<<<<<<<<<<<<<
    # concatinating two dataframes converts int values to floats
    # Therefore, we have to modify the float values back to int values as follows
cols = final.loc[:,'3d':'zip codes'].columns.values

print(len(cols), len(final))

for col in cols:
    final[col] = final[col].astype(int)

    # drop index column as it is not needed
final = final.drop(['index'], axis=1)

    #print(final.head())

    # write the dataframe to a csv file
final.to_csv("final_data.csv", index=False)


JSONToCSV()
