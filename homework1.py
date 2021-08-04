import csv
import  numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type(torch.FloatTensor)
class cvdataset(Dataset):
    def __init__(self,path):
        cvdataset.path=path
        fp=open(path,'r')
        dat=list(csv.reader(fp))
        dat=np.array(dat[1:])[:,1:].astype(float)
        self.label=torch.FloatTensor(dat[:,-1])
        self.data=torch.FloatTensor(dat[:,:93])
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)
        self.dim = self.data.shape[1]

    def __getitem__(self, idx):

        return self.data[idx],self.label[idx]


    def __len__(self):
        return len(self.label)
a=cvdataset('covid.train.csv')
class cvdtestset(Dataset):
    def __init__(self,path):
        cvdataset.path = path
        fp = open(path, 'r')
        dat = list(csv.reader(fp))
        dat = np.array(dat[1:])[:, 1:].astype(float)
        self.data = dat[:, :93]
        self.data=torch.FloatTensor(self.data)
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
        / self.data[:, 40:].std(dim=0, keepdim=True)
    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
b=cvdtestset('covid.test.csv')
train_dataloader=DataLoader(a,batch_size=270,shuffle=True)
test_dataloader=DataLoader(b,batch_size=1,shuffle=False)



class cvdNN(nn.Module):
    def __init__(self):
        super(cvdNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(93, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        a=self.net(x.float()).squeeze(1).float()
        return a

model=cvdNN()

epochs=2000

loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size=len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):


        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):

    res=[]
    with torch.no_grad():
        for X in dataloader:
            pred = model(X)
            res.append(pred)
    return res



for i in range(epochs):
    print(f"Epoch{i+1}\n-------------------------------")
    train(train_dataloader,model,loss_fn,optimizer)
model.eval()
data=test_loop(test_dataloader,model,loss_fn)
realdata=[]








for i in range(len(data)):
    realdata.append(data[i].numpy())

rrealdata=[]
for i in  range(len(realdata)):
    rrealdata.append([i,realdata[i][0]])


head=['id','tested_positive']
with open('data.csv','w') as f:
    f_csv=csv.writer(f)
    f_csv.writerow(head)
    for i in  range(len(rrealdata)):
        f_csv.writerow(rrealdata[i])
print('done!')