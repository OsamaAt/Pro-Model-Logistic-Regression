import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 

x_train=np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y_train=np.array([0,0,0,1,1,1,1,1,1,1])
x_test=np.array([[4.5]])

scaler=StandardScaler()
x_train_scaler=scaler.fit_transform(x_train)
x_test_scaler=scaler.transform(x_test)


model=LogisticRegression()
model.fit(x_train_scaler , y_train)
y_pred=model.predict(x_test_scaler)
print(f'If The Student Studied 4.5 hour(s) : {y_pred[0]:.4f}')
print('1 Succus✅ , 0 Fail❌')

x_fit=np.linspace(1 , 10 , 100).reshape(-1,1)
x_fit_norm=scaler.transform(x_fit)
y_fit=model.predict(x_fit_norm)

plt.scatter(x_train , y_train , color='red' , label='Data Training')
plt.plot(x_fit , y_fit , color='blue' , label='Data Fitting')
plt.legend()
plt.show()