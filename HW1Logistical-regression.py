import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from scipy import stats
df = pd.read_csv (r'C:\Users\jacso\Desktop\Homework\XLM-USD-Daily-09.15.14-10.30.21.csv')
df=df.dropna()
df.reset_index(drop=True, inplace=True)
print(df[df["Open"].isna()])

df.loc[df['Volume'] < 991699000, 'Volume'] = 0
df.loc[df['Volume'] > 991699000, 'Volume'] = 1
corrMatrix = df.corr()
#sn.heatmap(corrMatrix, annot=True)
print(df)
y = pd.DataFrame(df, columns= ['Volume'])
X = pd.DataFrame(df, columns= ['High'])


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=99)
model = LogisticRegression(solver='liblinear', random_state=0)
model = LogisticRegression(solver='liblinear', random_state=0).fit(X, y.values.ravel())

#model = LogisticRegression(solver='liblinear', C=10.0, random_state=99)
#model.fit(X, y.values.ravel())

print(model.intercept_)
print(model.coef_)
print(model.predict(X))
print(model.score(X, y))
print(confusion_matrix(y, model.predict(X)))
cm = confusion_matrix(y, model.predict(X))
print(classification_report(y, model.predict(X)))



plt.plot(X,y,'o', color='red')
#plt.plot(X, y)

