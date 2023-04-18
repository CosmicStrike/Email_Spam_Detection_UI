# %% [markdown]
# ## Email Spam detection using NLP

# %%
import pandas as pd
import re
import seaborn as sns

# %%
df = pd.read_csv("Dataset/spam.csv",encoding="ISO-8859-1")


# %%
df.drop(columns="Unnamed: 2",inplace=True, axis=1)
df.drop(columns="Unnamed: 3",inplace=True, axis=1)
df.drop(columns="Unnamed: 4",inplace=True, axis=1)

# %%
print(df.head())
print(df.describe())

# %%
# re_pattren = r'\b[A-Za-z0-9]+\b'

# def RE(s:str) -> str:
#     a = re.findall(re_pattren,s)
#     ans = ' '.join(a)
#     return ans

# %%
# l = []
# for i in range(df["MESSAGE"].count()):
#     l.append(RE(df["MESSAGE"][i]))

# %%
# df["Lvl1"] = l

# %%
# Perform Lemmation
import nltk
# nltk.download("wordnet") --- lemmatization -- downloaded
# nltk.download("punkt") --- tokenizor downloaded
# nltk.download("stopword") -- downloaded
# nltk.download('omw-1.4') - --lemmatization - -downloaded

# %%
def Calculate_Length(df):
    leng = []
    for i in range(df["v2"].count()):
        leng.append(len(df["v2"][i]))
    # print(leng)
    df["Length"] = leng
Calculate_Length(df)

# %%
sns.distplot(a=df[df['v1'] == "ham"].Length, kde=False)


# %%
sns.distplot(a=df[df['v1'] == "spam"].Length, kde=False)


# %%
from nltk import word_tokenize
def Tokenize(df):
    tokens = []
    for i in range(df["v2"].count()):
        tokens.append(word_tokenize(df["v2"][i]))
    df["tokens"] = tokens
Tokenize(df)
print("Original Text -> ", df["v2"][4])
print("Tokenized Text -> ", df["tokens"][4])

# %%
from nltk.corpus import stopwords
def RemoveStopword(df):
    stop_words = set(stopwords.words("english"))
    fil = []
    filters = []
    for i in range(df["tokens"].count()):
        fil = []
        for word in df["tokens"][i]:
            if word.casefold() not in stop_words:
                fil.append(word)
        filters.append(fil)
    df["Filtered-Stopwords"] = filters

RemoveStopword(df)

print("Tokenized Text -> ", df["tokens"][4])
print("After removing Stopwords -> ", df["Filtered-Stopwords"][4])


# %%
lem = nltk.WordNetLemmatizer()
def Lemmantize_Text(df):
    lemmantized = []
    l = []
    for i in range(df["Filtered-Stopwords"].count()):
        l = []
        for word in df["Filtered-Stopwords"][i]:
            l.append(lem.lemmatize(word))
        lemmantized.append(l)

    df["Lemmanted"] = lemmantized

Lemmantize_Text(df)
print("After removing Stopwords -> ", df["Filtered-Stopwords"][4])
print("Lemmatization -> ", df["Lemmanted"][4])

# %%
def FinalizeText(df):
    text = []
    for i in range(df["Lemmanted"].count()):
        t = ' '.join(df["Lemmanted"][i])
        text.append(t)
    df["Final Text"] = text
    # df.head()

FinalizeText(df)
print("Original text -> ",df["v2"][4])
print("Processed text -> ",df["Final Text"][4])


# %%
for i in range(df["v1"].count()):
    if df["v1"][i] == "ham":
        df["v1"][i] = int(0)
    else:
        df["v1"][i] = int(1)

df.head()

# %%
def RemoveUnwantedColumn(df):
    # print(df.head())
    df.drop(['tokens', 'Filtered-Stopwords','Lemmanted'], axis=1, inplace=True)
RemoveUnwantedColumn(df)


# %%
from sklearn.model_selection import train_test_split

y = pd.DataFrame(df["v1"])
df.drop(columns="v1", axis=1, inplace=True)
x = df
x_train, x_val, y_train, y_val = train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)


# %%
x_val.head()

# %%
# Count vertorization
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

cv = CountVectorizer(max_features=5000)
temp_train = cv.fit_transform(x_train['Final Text']).toarray()
temp_val = cv.transform(x_val['Final Text']).toarray()


# %%
# countVect= CountVectorizer()
# text = ["The dark flames of elixer becomes more dark - a monotonous"]
# cv = countVect.fit_transform(text).toarray()
# print(cv[0])
# print("Totol words are 12")
# print("Length of array is ", len(cv[0]))

# %%
# tf = TfidfTransformer()
# transform = tf.fit_transform(cv)
# print("TF-IDF : ",transform.todense())
# print("Count Vector :",cv[0])

# %%
# TF-IDF
tf = TfidfTransformer()
temp_train = tf.fit_transform(temp_train)
temp_val = tf.transform(temp_val)


# %%
#merging temp datafram with original dataframe
temp_train = pd.DataFrame(temp_train.toarray(), index=x_train.index)
temp_val = pd.DataFrame(temp_val.toarray(), index=x_val.index)
x_train = pd.concat([x_train, temp_train], axis=1, sort=False)
x_val = pd.concat([x_val, temp_val], axis=1, sort=False)

x_train.head()

# %%
#dropping the final_text column
x_train.drop(['Final Text'], axis=1, inplace=True)
x_val.drop(['Final Text'], axis=1, inplace=True)

x_train.head()

# %%
#dropping the v2 column
x_train.drop(['v2'], axis=1, inplace=True)
x_val.drop(['v2'], axis=1, inplace=True)

# %%
x_train.head()

# %%
#converting the labels to int datatype (for model training)
y_train = y_train.astype(int)
y_val = y_val.astype(int)


# %%
def Vectorize_And_TFIDFTranform(df):
    cv = CountVectorizer(max_features=5000)
    temp_train = cv.fit_transform(df['Final Text']).toarray()
    # print("temp_train ", temp_train)
    tf = TfidfTransformer()
    temp_train = tf.fit_transform(temp_train)
    # print("temp_train ", temp_train)

    temp_train = pd.DataFrame(temp_train.toarray())
    # print("temp_train ", temp_train)

    df = pd.concat([df, temp_train], axis=1, sort=False)
    # print(df)
    df.drop(['Final Text'], axis=1, inplace=True)
    df.drop(['v2'], axis=1, inplace=True)
    # print(df)
    return df

# %%
def PreProcess_Data(dataf):
    Calculate_Length(dataf)
    Tokenize(dataf)
    RemoveStopword(dataf)
    Lemmantize_Text(dataf)
    FinalizeText(dataf)
    RemoveUnwantedColumn(dataf)
    return Vectorize_And_TFIDFTranform(dataf)
    
