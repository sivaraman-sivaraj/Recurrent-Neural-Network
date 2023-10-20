import os 
import torch 
import numpy as np 
import torch.nn as nn
import matplotlib.pyplot as plt 
####################################
####################################
X_test0 = np.load("X_test.npy")
Y_test0 = np.load("Y_test.npy")

class RNN(nn.Module):
    def __init__(self,ip_length = 5, input_size = 1, hidden_size = 64, out_size = 1):
        super(RNN,self).__init__()
        self.ip_length   = ip_length
        self.batch_size  = 64
        self.hidden_size = hidden_size
        self.rnn         = nn.RNN(input_size, hidden_size,1)
        self.linear      = nn.Linear(hidden_size*ip_length,out_size)
        self.h0          = torch.zeros(1,ip_length,hidden_size)
    
    def forward(self,ip):
        op, self.hidden = self.rnn(ip, self.h0)
        pred = self.linear(torch.reshape(op,[self.batch_size,self.hidden_size*self.ip_length]))
        return pred 
    
net        = RNN() 
net.load_state_dict(torch.load("RNN_model.pt"))
net.eval() 
####################################################
####################################################
def Test_Data_maker(XX,YY):
    Xsample,Ysample = [],[]
    for i in range(len(XX)):
        Xsample.append(np.array(XX[i]).reshape((5,1)).tolist())
        Ysample.append(YY[i].tolist())
    return Xsample, Ysample

X_test1,Y_test1 = Test_Data_maker(X_test0,Y_test0) 
X_test = torch.tensor(X_test1)
Y_test = Y_test1[0:960] # torch.tensor(Y_test1) 
# print(X_test.shape)
# print(Y_test.shape) 
####################################################
#################################################### 
# ip = torch.rand(64,5,1)
Y_pred = []
for i in range(0,897,64):
    ip = X_test[i:i+64]
    op = net(ip)
    temp = op.reshape(1,64).tolist()
    Y_pred.extend(temp[0])

print(len(Y_pred))
print(len(Y_test))

plt.figure(figsize=(9,6))
plt.scatter(Y_pred,Y_test)
plt.grid()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



