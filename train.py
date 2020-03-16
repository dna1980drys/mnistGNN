import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from torch_geometric.nn import GCNConv
from torch_scatter import  scatter_max
import matplotlib.pyplot as plt
from load_mnist_graph import load_mnist_graph

data_size = 60000
train_size = 50000
batch_size = 100
epoch_num = 150

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(2, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 48)
        self.conv4 = GCNConv(48, 64)
        self.conv5 = GCNConv(64, 96)
        self.conv6 = GCNConv(96, 128)
        self.linear1 = torch.nn.Linear(128,64)
        self.linear2 = torch.nn.Linear(64,10)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = self.conv6(x, edge_index)
        x = F.relu(x)
        x, _ = scatter_max(x, data.batch, dim=0)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


def main():
    #前準備
    mnist_list = load_mnist_graph(data_size=data_size)
    device = torch.device('cuda')
    model = Net().to(device)
    trainset = mnist_list[:train_size]
    optimizer = torch.optim.Adam(model.parameters())
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = mnist_list[train_size:]
    testloader = DataLoader(testset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    history = {
        "train_loss": [],
        "test_loss": [],
        "test_acc": []
    }

    print("Start Train")
    
    #学習部分
    model.train()
    for epoch in range(epoch_num):
        train_loss = 0.0
        for i, batch in enumerate(trainloader):
            batch = batch.to("cuda")
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs,batch.t)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.cpu().item()
            if i % 10 == 9:
                progress_bar = '['+('='*((i+1)//10))+(' '*((train_size//100-(i+1))//10))+']'
                print('\repoch: {:d} loss: {:.3f}  {}'
                        .format(epoch + 1, loss.cpu().item(), progress_bar), end="  ")

        print('\repoch: {:d} loss: {:.3f}'
            .format(epoch + 1, train_loss / (train_size / batch_size)), end="  ")
        history["train_loss"].append(train_loss / (train_size / batch_size))

        correct = 0
        total = 0
        batch_num = 0
        loss = 0
        with torch.no_grad():
            for data in testloader:
                data = data.to(device)
                outputs = model(data)
                loss += criterion(outputs,data.t)
                _, predicted = torch.max(outputs, 1)
                total += data.t.size(0)
                batch_num += 1
                correct += (predicted == data.t).sum().cpu().item()

        history["test_acc"].append(correct/total)
        history["test_loss"].append(loss.cpu().item()/batch_num)
        endstr = ' '*max(1,(train_size//1000-39))+"\n"
        print('Test Accuracy: {:.2f} %%'.format(100 * float(correct/total)), end='  ')
        print(f'Test Loss: {loss.cpu().item()/batch_num:.3f}',end=endstr)


    print('Finished Training')

    #最終結果出力
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += data.t.size(0)
            correct += (predicted == data.t).sum().cpu().item()
    print('Accuracy: {:.2f} %%'.format(100 * float(correct/total)))

if __name__=="__main__":
    main()
