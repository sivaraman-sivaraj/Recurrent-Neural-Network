import os 
import numpy as np 
import matplotlib.pyplot as plt 
import torch.optim as optim
import torch 
import torch.nn as nn 
from sklearn.utils import shuffle
#################################
#################################

def sample_maker(X_train,Y_train,n,L):
    sample_id = np.random.choice(L,n) 
    # print(sample_id)
    Xsample,Ysample = [],[]
    for i in sample_id:
        Xsample.append(np.array(X_train[i]).reshape((5,1)).tolist())
        Ysample.append(Y_train[i].tolist())
    return Xsample, Ysample

Xp = np.load("X_all.npy")
Yp = np.load("Y_all.npy") 
X,Y = shuffle(Xp,Yp) 

X_train,X_val,X_test = X[0:int(0.7*len(X))],X[int(0.7*len(X)):int(0.9*len(X))],X[int(0.9*len(X)):]
Y_train,Y_val,Y_test = Y[0:int(0.7*len(X))],Y[int(0.7*len(X)):int(0.9*len(X))],Y[int(0.9*len(X)):]
np.save("X_test.npy",X_test)
np.save("Y_test.npy",Y_test)
######################################################
######################################################

class LSTM(nn.Module):
    def __init__(self,ip_length = 5, input_size = 1, hidden_size = 64, out_size = 1):
        super(LSTM,self).__init__()
        self.ip_length   = ip_length
        self.batch_size  = 64
        self.hidden_size = hidden_size
        self.lstm        = nn.LSTM(input_size, hidden_size,1)
        self.linear      = nn.Linear(self.hidden_size*self.ip_length,out_size)
        self.hidden      = (torch.zeros(1,ip_length,hidden_size),torch.zeros(1,ip_length,hidden_size))
    
    def forward(self,ip):
        lstm_out, self.hidden = self.lstm(ip, self.hidden)
        pred                  = self.linear(torch.reshape(lstm_out,[self.batch_size,self.hidden_size*self.ip_length]))
        return pred
    
######################################################
######################################################
net        = LSTM() 
optimizer  = torch.optim.Adam(net.parameters(),lr=1e-3)
criterion  = nn.MSELoss() #size_average = False
save_dir   = os.getcwd()
DP_count   = len(X_train)
DP_val_count = len(X_val)
No_Epoches   = 10000
######################################################
######################################################
def Train_RNN(X_train,Y_train, Epoches):
    Loss_train        = list()
    Loss_val          = list()
    for i in range(Epoches):
        x_sample,y_sample         = sample_maker(X_train,Y_train,64,DP_count)
        x_val_sample,y_val_sample = sample_maker(X_val,Y_val,64,DP_val_count)
        net.hidden    = (torch.zeros(1,5,net.hidden_size),torch.zeros(1,5,net.hidden_size))
        xt_sample     = torch.tensor(x_sample)
        yt_sample     = torch.tensor(y_sample) 
        x_val_sample  = torch.tensor(x_val_sample)
        y_val_sample  = torch.tensor(y_val_sample)
        #############
        y_train_pred = net(xt_sample) 
        loss_train   = criterion(y_train_pred,yt_sample)
        optimizer.zero_grad()
        loss_train.backward(retain_graph = True)
        optimizer.step()
        #############
        net.hidden    = (torch.zeros(1,5,net.hidden_size),torch.zeros(1,5,net.hidden_size))
        y_val_pred    = net(x_val_sample) 
        loss_val      = criterion(y_val_pred,y_val_sample) 
        #############
        Loss_train.append(loss_train.item()) 
        Loss_val.append(loss_val.item())
        ###########################################
        if (i%100) == 0:
            print('epoch {}, loss {}'.format(i, loss_train.item()))
    # print(xt_sample.shape)
    # print(yt_sample.shape)
    torch.save(net.state_dict(), save_dir+"\\Lstm_model.pt")  
    return Loss_train, Loss_val

Loss_rnn,Loss_val = Train_RNN(X_train,Y_train,No_Epoches)
np.save("loss_lstm.py",Loss_rnn)


plt.figure(figsize=(9,6))
plt.plot(Loss_rnn,color="teal",linewidth=2.0, label="Train Loss")
plt.plot(Loss_val,color="crimson",linewidth=2.0, label = "Val Loss")
plt.xlabel("Epoches")
plt.ylabel("MSE")
plt.legend(loc="best")
plt.grid()
plt.savefig("F1_lstm.jpg",dpi=300)
plt.show()
######################################################
######################################################
# RNN_net = RNN()
# ip = torch.rand(32,5,1) 
# h0 = torch.randn(1,5,64)
# op = RNN_net(ip) 
# print(op.shape)






