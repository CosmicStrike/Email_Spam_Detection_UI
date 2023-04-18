from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# n_estimator - number of trees (decesion tree) in forest
model = RandomForestClassifier(n_estimators=100)
print(type(y_train))
model.fit(x_train, y_train)
y_preds = model.predict(x_val.values)
print("Random Forest:", accuracy_score(y_val, y_preds))

text_df = pd.DataFrame(columns=["v2"])
text = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18'"]

text_df["v2"] = text
print(text_df)
text = PreProcess_Data(text_df)
print("TEXT _> ")
print(text)

# %%

# %%
def FitData(dataf,size):
    for i in range(dataf.shape[1],size+1):
        dataf[f"{i}"] = 0.0
FitData(text,5000)
text.head()

# %%
ans = model.predict(text)
print(ans)
if ans:
    print("Spam")
else:
    print("Ham")
