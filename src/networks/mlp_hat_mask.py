import sys
import torch
import torch.nn.init as init

import utils
class Linear_(torch.nn.Module):
    def __init__(self, in_neurons, out_neurons, bias=True, weight_initializer=None):
        super(Linear_, self).__init__()
        self.weight_initializer = weight_initializer
        self.weight = torch.nn.Parameter(torch.Tensor(out_neurons, in_neurons))
        if self.weight_initializer == "xavier":
            init.xavier_uniform_(self.weight)  # Xavier initialization for weights
        self.bias = torch.nn.Parameter(torch.zeros(out_neurons))

    def forward(self, x, mask):
        weights = self.weight * mask
        bias = self.bias * mask.max(dim=1)[0]
        return torch.nn.functional.linear(x, weights, bias)


class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla,nlayers=2,nhid=2000,pdrop1=0.2,pdrop2=0.5,args=None):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla

        self.nlayers=nlayers
        self.weight_initializer = args.weight_initializer

        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(pdrop1)
        self.drop2=torch.nn.Dropout(pdrop2)
        self.fc1=torch.nn.Linear(ncha*size*size,nhid)

        if self.weight_initializer == "xavier":
            init.xavier_uniform_(self.fc1.weight)  # Xavier initialization for weights

        self.efc1=torch.nn.Embedding(len(self.taskcla),nhid)
        if nlayers>1:
            self.fc2=torch.nn.Linear(nhid,nhid)
            if self.weight_initializer == "xavier":
                init.xavier_uniform_(self.fc2.weight)  # Xavier initialization for weights

            self.efc2=torch.nn.Embedding(len(self.taskcla),nhid)
            if nlayers>2:
                self.fc3=torch.nn.Linear(nhid,nhid)
                if self.weight_initializer == "xavier":
                    init.xavier_uniform_(self.fc3.weight)  # Xavier initialization for weights

                self.efc3=torch.nn.Embedding(len(self.taskcla),nhid)
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            linear_layer = torch.nn.Linear(nhid,n)
            if self.weight_initializer == "xavier":
                init.xavier_uniform_(linear_layer.weight)  # Xavier initialization for weights

            self.last.append(linear_layer)


        self.gate=torch.nn.Sigmoid()
        """ (e.g., used with compression experiments)
        lo,hi=0,2
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        self.efc3.weight.data.uniform_(lo,hi)
        #"""

        return

    def forward(self,t,x,s=1):
        # Gates
        masks=self.mask(t,s=s)
        if self.nlayers==1:
            gfc1=masks
        elif self.nlayers==2:
            gfc1,gfc2=masks
        elif self.nlayers==3:
            gfc1,gfc2,gfc3=masks
        # Gated
        h=self.drop1(x.view(x.size(0),-1))
        h=self.drop2(self.relu(self.fc1(h)))
        h=h*gfc1.expand_as(h)
        if self.nlayers>1:
            h=self.drop2(self.relu(self.fc2(h)))
            h=h*gfc2.expand_as(h)
            if self.nlayers>2:
                h=self.drop2(self.relu(self.fc3(h)))
                h=h*gfc3.expand_as(h)
        y=[]
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        return y,masks

    def mask(self,t,s=1):
        gfc1=self.gate(s*self.efc1(t))
        if self.nlayers==1: return gfc1
        gfc2=self.gate(s*self.efc2(t))
        if self.nlayers==2: return [gfc1,gfc2]
        gfc3=self.gate(s*self.efc3(t))
        return [gfc1,gfc2,gfc3]

    def get_view_for(self,n,masks):
        if self.nlayers==1:
            gfc1=masks
        elif self.nlayers==2:
            gfc1,gfc2=masks
        elif self.nlayers==3:
            gfc1,gfc2,gfc3=masks
        if n=='fc1.weight':
            return gfc1.data.view(-1,1).expand_as(self.fc1.weight)
        elif n=='fc1.bias':
            return gfc1.data.view(-1)
        elif n=='fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.fc2.weight)
            return torch.min(post,pre)
        elif n=='fc2.bias':
            return gfc2.data.view(-1)
        elif n=='fc3.weight':
            post=gfc3.data.view(-1,1).expand_as(self.fc3.weight)
            pre=gfc2.data.view(1,-1).expand_as(self.fc3.weight)
            return torch.min(post,pre)
        elif n=='fc3.bias':
            return gfc3.data.view(-1)
        return None

