import torch

import utils
import torch.nn.init as init


class Conv2D_(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, weight_initializer=None):
        super(Conv2D_, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.weight_initializer = weight_initializer
        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if self.weight_initializer == "xavier":
            init.xavier_uniform_(self.weight)  # Xavier initialization for weights

        self.bias = torch.nn.Parameter(torch.zeros(out_channels))
    # def ini..()
    def forward(self, x, mask):
        weights = self.weight * mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
        bias = self.bias * mask.max(dim=1)[0]
        return torch.nn.functional.conv2d(x, weights, bias, self.stride, self.padding, self.dilation, self.groups)


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

    def __init__(self, inputsize, taskcla,args=None):
        super(Net, self).__init__()

        ncha, size, _ = inputsize
        self.ncha = ncha
        self.taskcla = taskcla
        self.args = args
        self.weight_initializer = args.weight_initializer

        self.c1 = Conv2D_(ncha, 64, kernel_size=size // 8,weight_initializer=self.weight_initializer)
        s = utils.compute_conv_output_size(size, size // 8)
        s = s // 2
        self.c2 = Conv2D_(64, 128, kernel_size=size // 10, weight_initializer=self.weight_initializer)
        s = utils.compute_conv_output_size(s, size // 10, )
        s = s // 2
        self.c3 = Conv2D_(128, 256, kernel_size=2, weight_initializer=self.weight_initializer)
        s = utils.compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()

        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)
        self.fc1 = Linear_(256 * self.smid * self.smid, 2048, weight_initializer=self.weight_initializer)
        self.fc2 = Linear_(2048, 2048, weight_initializer=self.weight_initializer)
        self.last = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(torch.nn.Linear(2048, n))

        self.gate = torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        self.ec1 = torch.nn.Embedding(len(self.taskcla), ncha * 64)
        self.ec2 = torch.nn.Embedding(len(self.taskcla), 64 * 128)
        self.ec3 = torch.nn.Embedding(len(self.taskcla), 128 * 256)
        self.efc1 = torch.nn.Embedding(len(self.taskcla), 1024 * 2048)
        self.efc2 = torch.nn.Embedding(len(self.taskcla), 2048 * 2048)
        # """ (e.g., used in the compression experiments)
        lo,hi=0,2 ##
        self.ec1.weight.data.uniform_(lo,hi)
        self.ec2.weight.data.uniform_(lo,hi)
        self.ec3.weight.data.uniform_(lo,hi)
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        #"""

        return

    def forward(self, t, x, s=1):
        # Gates
        masks = self.mask(t, s=s)
        gc1, gc2, gc3, gfc1, gfc2 = masks
        # Gated
        h = self.maxpool(self.drop1(self.relu(self.c1(x, gc1))))
        # h = h * gc1.view(1, -1, 1, 1).expand_as(h)
        h = self.maxpool(self.drop1(self.relu(self.c2(h, gc2))))
        # h = h * gc2.view(1, -1, 1, 1).expand_as(h)
        h = self.maxpool(self.drop2(self.relu(self.c3(h, gc3))))
        # h = h * gc3.view(1, -1, 1, 1).expand_as(h)
        h = h.view(x.size(0), -1)
        h = self.drop2(self.relu(self.fc1(h, gfc1)))
        # h = h * gfc1.expand_as(h)
        h = self.drop2(self.relu(self.fc2(h, gfc2)))
        # h = h * gfc2.expand_as(h)
        y = []
        for i, _ in self.taskcla:
            y.append(self.last[i](h))
        return y, masks

    def mask(self, t, s=1):
        ## ss
        gc1 = self.gate(s * self.ec1(t).view(64, 3))
        # if self.args.experiment == "cifar":
        #     gc1 = self.gate(s * self.ec1(t).view(64,1))
        # else:
        gc1 = self.gate(s * self.ec1(t).view(64, self.ncha))



        gc2 = self.gate(s * self.ec2(t).view(128, 64))
        gc3 = self.gate(s * self.ec3(t).view(256, 128))
        gfc1 = self.gate(s * self.efc1(t).view(2048, 1024))
        gfc2 = self.gate(s * self.efc2(t).view(2048, 2048))
        return [gc1, gc2, gc3, gfc1, gfc2]

    # def get_view_for(self, n, masks):
    #     gc1, gc2, gc3, gfc1, gfc2 = masks
    #     if n == 'fc1.weight':
    #         post = gfc1.data.view(-1, 1).expand_as(self.fc1.weight)
    #         pre = gc3.data.view(-1, 1, 1).expand((self.ec3.weight.size(1), self.smid, self.smid)).contiguous().view(1,
    #                                                                                                                 -1).expand_as(
    #             self.fc1.weight)
    #         return torch.min(post, pre)
    #     elif n == 'fc1.bias':
    #         return gfc1.data.view(-1)
    #     elif n == 'fc2.weight':
    #         post = gfc2.data.view(-1, 1).expand_as(self.fc2.weight)
    #         pre = gfc1.data.view(1, -1).expand_as(self.fc2.weight)
    #         return torch.min(post, pre)
    #     elif n == 'fc2.bias':
    #         return gfc2.data.view(-1)
    #     elif n == 'c1.weight':
    #         return gc1.data.view(-1, 1, 1, 1).expand_as(self.c1.weight)
    #     elif n == 'c1.bias':
    #         return gc1.data.view(-1)
    #     elif n == 'c2.weight':
    #         post = gc2.data.view(-1, 1, 1, 1).expand_as(self.c2.weight)
    #         pre = gc1.data.view(1, -1, 1, 1).expand_as(self.c2.weight)
    #         return torch.min(post, pre)
    #     elif n == 'c2.bias':
    #         return gc2.data.view(-1)
    #     elif n == 'c3.weight':
    #         post = gc3.data.view(-1, 1, 1, 1).expand_as(self.c3.weight)
    #         pre = gc2.data.view(1, -1, 1, 1).expand_as(self.c3.weight)
    #         return torch.min(post, pre)
    #     elif n == 'c3.bias':
    #         return gc3.data.view(-1)
    #     return None

    def get_view_for(self, n, masks):
        gc1, gc2, gc3, gfc1, gfc2 = masks
        if n == 'fc1.weight':
            # post = gfc1.data.view(-1, -1).expand_as(self.fc1.weight)

            post = gfc1.data.view(2048, 1024).expand_as(self.fc1.weight)
            # pre = gc3.data.view(-1, 1, 1).expand((self.ec3.weight.size(1), self.smid, self.smid)).contiguous().view(
            #     1, -1).expand_as(self.fc1.weight)
            # return torch.min(post, pre)
            return post
        elif n == 'fc1.bias':
            # return gfc1.data.view(-1)
            return gfc1.data.max(dim=1)[0]

        elif n == 'fc2.weight':
            post = gfc2.data.view(2048, 2048).expand_as(self.fc2.weight)
            # pre = gfc1.data.view(1, -1).expand_as(self.fc2.weight)
            # return torch.min(post, pre)
            return post
        elif n == 'fc2.bias':
            # return gfc2.data.view(-1)
            return gfc2.data.max(dim=1)[0]
        elif n == 'c1.weight':
            return gc1.data.view(64, 3, 1, 1).expand_as(self.c1.weight)
        elif n == 'c1.bias':
            # return gc1.data.view(-1)
            return gc1.data.max(dim=1)[0]
        elif n == 'c2.weight':
            post = gc2.data.view(128, 64, 1, 1).expand_as(self.c2.weight)

            # pre = gc1.data.view(64, 3, 1, 1).expand_as(self.c2.weight)
            # return torch.min(post, pre)
            return post

        elif n == 'c2.bias':
            # return gc2.data.view(-1)
            return gc2.data.max(dim=1)[0]

        elif n == 'c3.weight':
            post = gc3.data.view(256, 128, 1, 1).expand_as(self.c3.weight)
            # pre = gc2.data.view(128, 64, 1, 1).expand_as(self.c3.weight)
            # return torch.min(post, pre)
            return post
        
        elif n == 'c3.bias':
            # return gc3.data.view(-1)
            return gc3.data.max(dim=1)[0]

        return None