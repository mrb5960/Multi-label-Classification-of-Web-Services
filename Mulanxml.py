import pandas as pd

#df = pd.read_csv("tok&lem_final_data_v2_tfidf_renamedLabels.csv")
df = pd.read_csv("tok&lem_final_data_v3_top20labels_tfidf_500f.csv")

#newdf = df.loc[:,'threeD':'zip codes']
# 10 labels
#newdf = df.loc[:,'l_analytics':'l_tools']
# 50 labels
#newdf = df.loc[:,'l_advertising':'l_voice']
# 20 labels
newdf = df.loc[:,'l_analytics':'l_video']

cols = newdf.columns.values

#print cols

xmlfile = open("mlc_20labels.xml", "w")

xmlfile.write("<?xml version=\"1.0\" encoding=\"utf-8\"?>")
xmlfile.write("<labels xmlns=\"http://mulan.sourceforge.net/labels\">")

count = 0

for e in cols:
    for ch in ['&',' ','-','(',')']:
        if ch in e:
            e = e.replace(ch,".")
    count += 1
    content = "\t<label name=\"" + e + "\"></label>\n"
    #print content
    xmlfile.write(content)

xmlfile.write("</labels>")
xmlfile.close()

print count

