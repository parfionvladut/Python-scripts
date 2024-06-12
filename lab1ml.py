import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from scipy import stats
df = pd.read_csv (r'C:\Users\jacso\Desktop\Homework\bd (1).csv')
#corrMatrix = df.corr()
#print (corrMatrix)
#sn.heatmap(corrMatrix, annot=True)
#plt.show()#regresie liniara heap abdomen si heap inaltime
X = pd.DataFrame(df, columns= ['abdomen'])
y = pd.DataFrame(df, columns= ['forearm'])

#res = stats.linregress(X, y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
model = LinearRegression().fit(X, y)
y_pred= model.predict(X)
r_sq= model.score(X, y)
print('coefR^2:', r_sq)
print (y_test)
regressor=LinearRegression()
regressor.fit(X_train,y_train)#antrenarea algoritmului
print(regressor.intercept_)
print(regressor.coef_)
#compararea valorilor reale cu cele prezise
y_pred=regressor.predict(X_test)
print(X.shape)
mymodel = np.poly1d(np.polyfit(X, y, 3))

myline = np.linspace(1, 22, 100)
plt.plot(X,y,'o', color='red')
#plt.plot(X, regressor.coef_*X + regressor.intercept_)
#plt.plot(X, res.intercept + res.slope*X)
plt.plot(myline, mymodel(myline))

#CalculuiErorilor
print('Eroarea medie absoluta:',metrics.mean_absolute_error(y_test,y_pred))#MAE
print('Eroare medie patratica:',metrics.mean_squared_error(y_test,y_pred))#MSE
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))