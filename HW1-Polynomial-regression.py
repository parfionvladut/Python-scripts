import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
df = pd.read_csv (r'C:\Users\jacso\Desktop\Homework\XLM-USD-Daily-09.15.14-10.30.21.csv')
df=df.dropna()
df.reset_index(drop=True, inplace=True)
print(df[df["Open"].isna()])


corrMatrix = df.corr()
#sn.heatmap(corrMatrix, annot=True)
print(df)
y = pd.DataFrame(df, columns= ['Volume'])
X = pd.DataFrame(df, columns= ['High'])
y=y*256

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
model = LinearRegression().fit(X, y)
poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(X)
 
poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)


plt.plot(X,y,'o', color='red')
plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red')
plt.title('Linear Regression')
plt.xlabel('Value')
plt.ylabel('Volume')

#CalculuiErorilor

