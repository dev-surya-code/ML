import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('simple.csv')
dataset.head()
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtr,ytr)
ypr=model.predict(xte)
print(ypr)
print(yte)
plt.scatter(xtr,ytr,color='red')
plt.plot(xtr,model.predict(xtr),color='blue')
plt.title("mileage vs selling price(training)")
plt.xlabel("mileage")
plt.ylabel("selling price")
plt.show()
plt.scatter(xte,yte,color='red')
plt.plot(xte,model.predict(xte),color='blue')
plt.title("mileage vs selling price(testing)")
plt.xlabel("mileage")
plt.ylabel("selling price")
plt.show()
