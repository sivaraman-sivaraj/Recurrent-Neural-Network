import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def random_curve():
    Y = [] 
    X = np.linspace(-5,5,1000) 
    for i in range(len(X)):
        if X[i] < 1.256:

            y_temp = (np.exp(X[i]) + 0.5) + np.sin(2*np.pi*(i/100)*X[i]) 
        else:
            y_temp = (-np.log(X[i])+0.5)*10 + (np.sin(2*np.pi*(i/100)*X[i])*0.5)
      
        Y.append(y_temp)
    X_en,Y_en = [],[] 

    for j in range(len(X)-1):
        x_temp = np.linspace(X[j],X[j+1],10).tolist() 
        y_temp = np.linspace(Y[j],Y[j+1],10).tolist() 
        X_en.extend(x_temp)
        Y_en.extend(y_temp)
                   
    return np.array(X_en),np.array(Y_en)

X_,Y_ = random_curve() 
X_mean,X_std = np.mean(X_),np.std(X_)
Y_mean, Y_std = np.mean(Y_),np.std(Y_)

X = (X_-X_mean)/X_std 
Y = (Y_ - Y_mean)/Y_std 


X_tsv, Y_tsv = [],[]

for i in range(2,len(X)-2):
    x_temp = [X[i-2],X[i-1],X[i],X[i+1],X[i+2]]
    y_temp = [Y[i]] 
    X_tsv.append(x_temp)
    Y_tsv.append(y_temp) 

np.save("X_all.npy",X_tsv)
np.save("Y_all.npy",Y_tsv)


A = np.load("X_all.npy")
B = np.load("Y_all.npy") 

print(A.shape)
print(B.shape) 

plt.figure(figsize=(9,6))
plt.plot(X,Y,linewidth = 2.0) 
plt.grid() 
plt.show() 




