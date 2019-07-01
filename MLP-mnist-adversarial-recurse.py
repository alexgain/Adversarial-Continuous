import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F

from torchvision import datasets, transforms

#from fashion import fashion

import struct
from copy import deepcopy
from time import time, sleep
import gc

from sklearn.preprocessing import normalize

import argparse

np.random.seed(1)

cuda_boole = torch.cuda.is_available()


parser = argparse.ArgumentParser()
parser.add_argument('--N', type = int, default=60000)
parser.add_argument('--BS', type = int, default=128)
parser.add_argument('--save', type = int, default=0)
parser.add_argument('--no_train', type = int, default=1)
parser.add_argument('--no_train_all', type = int, default=0)
parser.add_argument('--loadD', type = int, default=0)
parser.add_argument('--N2', type = int, default=200)
parser.add_argument('--epochs', type = int, default=2000)
parser.add_argument('--LR', type = float, default=0.01)
parser.add_argument('--width', type = int, default=20)
parser.add_argument('--widthD', type = int, default=500)
parser.add_argument('--iters', type = int, default=12)
parser.add_argument('--recurse', type = int, default=4)
parser.add_argument('--fgsm', type = int, default=0)

args = parser.parse_args()

###                               ###
### Data import and preprocessing ###
###                               ###

N = args.N
BS = args.BS
save = args.save
no_train = args.no_train
no_train_all = args.no_train_all
loadD = args.loadD

N2 = args.N2
epochs = args.N2
LR = args.LR
width = args.width
widthD = args.widthD


transform_data = transforms.ToTensor()

train_set = datasets.MNIST('./data', train=True, download=True,
                   transform=transform_data)

# train_set.train_data = train_set.train_data[:N]

train_loader = torch.utils.data.DataLoader(train_set, batch_size = BS, shuffle=False)

train_loader_bap = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transform_data),
    batch_size=N2, shuffle=False)

test_dataset = datasets.MNIST(root='./data', 
                            train=False, 
                            transform=transform_data,
                           )
##adding noise to test:
##test_dataset.test_data = test_dataset.test_data.float()
##test_dataset.test_data = test_dataset.test_data + noise_level*torch.abs(torch.randn(*test_dataset.test_data.shape))
##test_dataset.test_data = test_dataset.test_data / test_dataset.test_data.max()

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=N2, shuffle=False)

test_loader_bap = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=N2, shuffle=False)


##adding noise to test:
##test_dataset.test_data = test_dataset.test_data.float()
##test_dataset.test_data = test_dataset.test_data + noise_level*torch.abs(torch.randn(*test_dataset.test_data.shape))
##test_dataset.test_data = test_dataset.test_data / test_dataset.test_data.max()

##test_loader = torch.utils.data.DataLoader(
##    datasets.MNIST('./data', train=False, transform=transforms.Compose([
##                       transforms.ToTensor(),
##                       transforms.Normalize((0.1307,), (0.3081,))
##                   ])),
##    batch_size=10000, shuffle=False)

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]
##ytrain = to_categorical(ytrain, 3)
##ytest = to_categorical(ytest, 3)

##entropy = np.load('entropy_mnist_train.npy')[:N2]
##entropy_test = np.load('entropy_mnist_test.npy')[:N2]
##np.random.shuffle(entropy)

###                      ###
### Define torch network ###
###                      ###

class Net(nn.Module):
    def __init__(self, input_size, width, num_classes):
        super(Net, self).__init__()

        ##feedfoward layers:

        self.ff1 = nn.Linear(input_size, width) #input

        self.ff2 = nn.Linear(width, width) #hidden layers
        self.ff3 = nn.Linear(width, width)
        self.ff4 = nn.Linear(width, width)
        self.ff5 = nn.Linear(width, width)

##        self.ff_out = nn.Linear(width, num_classes, bias = bias_ind) #output     
        self.ff_out = nn.Linear(width, num_classes) #output     
        
        ##activations:
        self.do = nn.Dropout()
        self.relu = nn.ReLU()
        self.sm = nn.Softmax()

        
    def forward(self, input_data):

        out = self.relu(self.ff1(input_data)) #input
        out = self.relu(self.ff2(out)) #hidden layers
        out = self.relu(self.ff3(out))
        out = self.relu(self.ff4(out))
        out = self.relu(self.ff5(out))
        out = self.ff_out(out)

        return out


class NetD(nn.Module):
    def __init__(self, input_size, width, num_classes):
        super(Net, self).__init__()

        ##feedfoward layers:

        self.ff1 = nn.Linear(input_size, width) #input

        self.ff2 = nn.Linear(width, width) #hidden layers
        self.ff3 = nn.Linear(width, width)
        self.ff4 = nn.Linear(width, width)
        self.ff5 = nn.Linear(width, width)

##        self.ff_out = nn.Linear(width, num_classes, bias = bias_ind) #output     
        self.ff_out = nn.Linear(width, num_classes) #output     
        
        ##activations:
        self.do = nn.Dropout()
        self.relu = nn.ReLU()
        self.sm = nn.Softmax()

        
    def forward(self, input_data):

        out = self.relu(self.ff1(input_data)) #input
        out = self.relu(self.ff2(out)) #hidden layers
        out = self.relu(self.ff3(out))
        out = self.relu(self.ff4(out))
        out = self.relu(self.ff5(out))
        out = self.ff_out(out)

        return out



###hyper-parameters:
input_size = 28*28
num_classes = 10

###defining network:        
my_net = Net(input_size, width, num_classes)
if no_train:
    my_net.load_state_dict(torch.load('trained_model.state'))
if cuda_boole:
    my_net = my_net.cuda()

# my_netD = Net(input_size, widthD, num_classes)
# if loadD:
#     # my_netD.load_state_dict(torch.load('defense_trained.state'))
#     my_netD = torch.load('defense_trained.state')
# if cuda_boole:
#     my_netD = my_netD.cuda()
    
recurse_nets = [Net(input_size, widthD, num_classes) for i in range(args.recurse)]
if cuda_boole:
    recurse_nets = [net.cuda() for net in recurse_nets]

###                       ###
### Loss and optimization ###
###                       ###

LR2 = 1.0
##loss_metric = nn.MSELoss()
loss_metric = nn.CrossEntropyLoss()
##loss_metric = torch.nn.NLLLoss()
# optimizer = torch.optim.SGD(my_net.parameters(), lr = LR, momentum = 0.9)
##optimizer = torch.optim.RMSprop(my_net.parameters(), lr = 0.00001)
##optimizer = torch.optim.RMSprop(my_net.parameters(), lr = 0.00001, momentum = 0.8)
optimizer = torch.optim.Adam(my_net.parameters(), lr = 0.001)
# optimizerD = torch.optim.Adam(my_netD.parameters(), lr = 0.001)
optimizerDr = [torch.optim.Adam(net.parameters(), lr = args.LR) for net in recurse_nets]

###                         ###
### Adversarial Attack code ###
###                         ###

class FGSM():
    def __init__(self, loss, epsilon):
        self.loss = loss
        self.epsilon = epsilon

    def forward(self, x, y_true, model, modelD=None):        
        
        x_adv = x

        y = model.forward(x_adv)
        if modelD is not None:
            for net in modelD:
                y += net(x_adv)
                
                
        J = self.loss(y,y_true)# - 1*bap_val(0)

        if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)

        x_grad = torch.autograd.grad(J, x_adv)[0]
        x_adv = x + self.epsilon*x_grad.sign_()
        x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv

class IFGSM():
    def __init__(self, loss, epsilon=0.3, iters=8, alpha=0):
        self.loss = loss
        self.epsilon = epsilon
        if iters == 0 :
            self.iters = int(min(epsilon*255 + 4, 1.25*epsilon*255))
        else:
            self.iters = iters
        if alpha == 0:
            self.alpha = epsilon/iters
        else:
            self.alpha = alpha

    def forward(self, x, y_true, model, modelD):
        x_adv = x
    
        for k in range(self.iters):
            x.requires_grad = True
            # x_adv.requires_grad = True
            y = model.forward(x)
            if modelD is not None:
                for net in modelD:
                    y += net(x_adv)

            J = self.loss(y,y_true)# - 1*bap_val(0)

            if x_adv.grad is not None:
                    x_adv.grad.data.fill_(0)

            # x_grad = torch.autograd.grad(J, x_adv)[0]
            x_grad = torch.autograd.grad(J, x)[0]
            x_adv = x + self.alpha*x_grad.sign_()

            ##clamping:
            a = torch.clamp(x - self.epsilon, min=0)
            b = (x_adv>=a).float()*x_adv + (a>x_adv).float()*a
            c = (b > x+self.epsilon).float()*(x+self.epsilon) + (x+self.epsilon >= b).float()*b
            x = torch.clamp(c, max=1).detach_()
            # x_adv = torch.clamp(x_adv, 0, 1) #0-1 clamping

        x_adv = x

        return x_adv



# adv_attack = FGSM(loss_metric, 0.3)
adv_attack = IFGSM(loss_metric, epsilon = 0.3, iters = args.iters, alpha=0.3)
if args.fgsm:
    adv_attack = FGSM(loss_metric, 0.3)

###                 ###
### Attractor Algs. ###
###                 ###

#Some more hyper-params and initializations:

eps_ball = 3.0 #controls how big we want the attractor spaces


###          ###
### Training ###
###          ###

#Some more hyper-params and initializations:

##train_loader = torch.utils.data.DataLoader(train, batch_size=BS, shuffle=True)
##test_loader = torch.utils.data.DataLoader(test, batch_size=BS, shuffle=False)

##printing train statistics:
def train_acc():
    correct = 0
    total = 0
    for images, labels in train_loader:
        if cuda_boole:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images.view(-1, 28*28))
        outputs = my_net(images)
        _, predicted = torch.max(outputs.data, 1)
    ##    labels = torch.max(labels.float(),1)[1]
    ##    predicted = torch.round(outputs.data).view(-1).long() 
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()

    print('Accuracy of the network on the train images: %f %%' % (100.0 * np.float(correct) / np.float(total)))
    
def test_acc():
    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        if cuda_boole:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images.view(-1, 28*28))
        outputs = my_net(images)
        _, predicted = torch.max(outputs.data, 1)
    ##    labels = torch.max(labels.float(),1)[1]
    ##    predicted = torch.round(outputs.data).view(-1).long()
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()

    print('Accuracy of the network on the 10000 test images: %f %%' % (100.0 * np.float(correct) / np.float(total)))

def test_acc_adv():
    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader_bap:
        if cuda_boole:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images.view(-1,28*28), requires_grad=True)
        images = adv_attack.forward(images, Variable(labels), my_net, recurse_nets)
        outputs = my_net(images)
        _, predicted = torch.max(outputs.data, 1)
    ##    labels = torch.max(labels.float(),1)[1]
    ##    predicted = torch.round(outputs.data).view(-1).long()
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()
        break

    print('Accuracy of the network on the 10000 test images: %f %%' % (100.0 * np.float(correct) / np.float(total)))

def test_acc_adv_def():
    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader_bap:
        if cuda_boole:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images.view(-1,28*28), requires_grad=True)
        images = adv_attack.forward(images, Variable(labels), my_net, recurse_nets)
        outputs = my_net(images)
        for net in recurse_nets:
            outputs += net(images)
        _, predicted = torch.max(outputs.data, 1)
    ##    labels = torch.max(labels.float(),1)[1]
    ##    predicted = torch.round(outputs.data).view(-1).long()
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()
        break

    print('Accuracy of the network on the 10000 test images: %f %%' % (100.0 * np.float(correct) / np.float(total)))

def test_acc_adv_def2():
    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader_bap:
        if cuda_boole:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images.view(-1,28*28), requires_grad=True)
        # images = adv_attack.forward(images, Variable(labels), my_net)
        outputs = my_net(images)
        for net in recurse_nets:
            outputs += net(images)
        _, predicted = torch.max(outputs.data, 1)
    ##    labels = torch.max(labels.float(),1)[1]
    ##    predicted = torch.round(outputs.data).view(-1).long()
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()
        break

    print('Accuracy of the network on the 10000 test images: %f %%' % (100.0 * np.float(correct) / np.float(total)))
    
train_acc()
test_acc()
test_acc_adv()

if no_train_all:
    for K in np.arange(1,1000,10):
        print('Iteration:',K)
        adv_attack = IFGSM(loss_metric, epsilon = 0.3, iters = K, alpha=0.3)
        test_acc()
        test_acc_adv()
        test_acc_adv_def()
        test_acc_adv_def2()        

###training loop (w/corr):
loss = 0
K1 = 0
t1 = time()
for epoch in range(epochs):

    # if epoch%10==0:
    #     K1 = (K1 + 1)%args.recurse
    # if K1 == 0:
    #     K1 = 1
    
    state_distance = 0

    ##time-keeping 1:
    time1 = time()

    for i, (x,y) in enumerate(train_loader):
        
        ##some pre-processing:
        x = x.view(-1,28*28)
##        y = y.float()
##        y = y.long()
##        y = torch.Tensor(to_categorical(y.long().cpu().numpy(),num_classes)) #MSE

        ##cuda:
        if cuda_boole:
            x = x.cuda()
            y = y.cuda()

        ##data preprocessing for optimization purposes:        
        x = Variable(x,requires_grad=True)
        x_adv = adv_attack.forward(x, Variable(y), my_net, recurse_nets)

        ###regular BP gradient update:
        if not no_train:
            optimizer.zero_grad()
        outputs = my_net.forward(x)
        outputs_adv = my_net.forward(x_adv)
        if not no_train:
            loss = loss_metric(outputs,y)
            loss.backward(retain_graph=True)
            optimizer.step()
        
        outputsr = [outputs]
        outputs_advr = [outputs_adv]
        # for K in range(K1):
        for K in range(args.recurse):
            outputsr.append(outputsr[K] + recurse_nets[K](x))    
            outputs_advr.append(outputs_advr[K] + recurse_nets[K](x_adv))    

        ##Defense network update 1:
        # for K in range(K1):
        for K in range(args.recurse):
            optimizerDr[K].zero_grad()
            lossD = ((outputs_advr[K+1] - (outputsr[K] - outputs_advr[K]))**2).mean()
            lossD.backward(retain_graph=True)
            optimizerDr[K].step()

        # for K in range(K1):
        for K in range(args.recurse):
            optimizerDr[K].zero_grad()
            lossD = ((outputsr[K+1] - 0)**2).mean()
            lossD.backward(retain_graph=True)
            optimizerDr[K].step()

        # for K in range(K1):
        #     optimizerDr[K].step()
            
            
        # for 

        ###Defense network update 1:
        # optimizerD.zero_grad()
        # outputsD = my_netD(x_adv)        
        # lossD = ((outputsD - (outputs - outputs_adv))**2).mean()
        # lossD.backward()
        
        # # for K in range(args.recurse):
        # #     optimizerDr[K].zero_grad()
        # #     outputsDr = recurse_nets[K](x_adv)
        # #     lossD = ((outputsD - (outputs - outputs_adv))**2).mean()
        # #     lossD.backward()

        # ###Defense network update 2:
        # outputsD = my_netD(x)        
        # lossD2 = ((outputsD - 0)**2).mean()
        # lossD2.backward()


        # optimizerD.step()        
                
        ##performing update:

        ##Performing attractor update:
        # rand_vec = Variable(torch.randn(*list(x.shape)))
        # if cuda_boole:
        #     rand_vec = rand_vec.cuda()
        # x_pert = x + (eps_ball)*(rand_vec / rand_vec.norm())
        
        # optimizer.zero_grad()

        # ##getting two states:
        # state1 = my_net.forward(x)
        # state2 = my_net.forward(x_pert)

        # loss2 = LR2*(state1 - state2).norm()
        # loss2.backward(retain_graph = True)
                
        ##performing update:
        # optimizer.step()

        ##accumulating loss:
        # state_distance += float(loss.cpu().data.numpy())
        
        ##printing statistics:
        
        ##printing statistics:
        if (i+1) % np.floor(N/BS) == 0:
            
            if not no_train:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                       %(epoch+1, epochs, i+1, N//BS, loss.data.item()))
            else:
                print ('Epoch [%d/%d], Step [%d/%d]' 
                       %(epoch+1, epochs, i+1, N//BS))
            # print('Avg Batch Distance:',state_distance/(i+1))

            train_acc()
            test_acc()
            test_acc_adv()
            test_acc_adv_def()
            test_acc_adv_def2()            
            print("Defense net minibatch loss:",lossD.data.item())

    ##time-keeping 2:
    time2 = time()
    print('Elapsed time for epoch:',time2 - time1,'s')
    print('ETA of completion:',(time2 - time1)*(epochs - epoch - 1)/60,'minutes')
    print()

t2 = time()
print((t2 - t1)/60,'total minutes elapsed')

if save == True:
    torch.save(my_net.state_dict(),'trained_model.state')