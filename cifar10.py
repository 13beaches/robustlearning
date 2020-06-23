import sys, os
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/..")

import argparse
from datetime import datetime
import numpy as np
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import torch
import torch.nn.functional as F
import deadZoneOptimizer as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from learningStats import learningStats
from RAdam_master.radam import radam
#import lookahead_pytorch as lookahead

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

class Network(torch.nn.Module):
    def __init__(self, robustHandler):
        super(Network, self).__init__()

        self.robustHandler = robustHandler

        # conv block
        self.conv = torch.nn.ModuleList([
            torch.nn.Conv2d(  3,  96, kernel_size=3, padding=1),
            torch.nn.Conv2d( 96,  96, kernel_size=3, padding=1),
            torch.nn.Conv2d( 96,  96, kernel_size=3, padding=1),
            torch.nn.Conv2d( 96, 192, kernel_size=3, padding=1),
            torch.nn.Conv2d(192, 192, kernel_size=3, padding=1),
            torch.nn.Conv2d(192, 192, kernel_size=3, padding=1),
            torch.nn.Conv2d(192, 192, kernel_size=3, padding=1),
            torch.nn.Conv2d(192, 192, kernel_size=1, padding=0),
            torch.nn.Conv2d(192, 192, kernel_size=1, padding=0),
        ])
        
        self.pool = torch.nn.ModuleList([
            torch.nn.MaxPool2d(2),
            torch.nn.MaxPool2d(2),
        ])

        self.drop = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()

        for conv in self.conv:
            self.bn.append(torch.nn.BatchNorm2d(conv.out_channels, affine=False))
            self.drop.append(torch.nn.Dropout(0.50))
        
        self.fc = torch.nn.ModuleList([
            torch.nn.Linear(192, 10),
        ])

        for conv in self.conv:
            self.robustHandler.register(conv)
        
        for fc in self.fc:
            self.robustHandler.register(fc)

    def forward(self, x):
        # x += torch.randn_like(x) * 0.15

        for i, (conv, bn) in enumerate(zip(self.conv, self.bn)):
            x = F.relu(bn(conv(x)))
            if i==2 or i==5:
                x = self.pool[i//3](x)
                x = self.drop[i//3](x)

        # print(x.shape)
        x = torch.mean(x, dim=(2, 3))
        x = self.fc[0](x)

        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', default=-1, type=int, help='Cuda device to use (-1 for none)')
    parser.add_argument('-gpu',   type=int,   default=[0], nargs='+', help='which gpu(s) to use')
    parser.add_argument('-b',     type=int,   default=128,      help='batch size for dataloader')
    parser.add_argument('-warm',  type=int,   default=1,        help='warm up training phase')
    parser.add_argument('-lr',    type=float, default=0.1,      help='initial learning rate')
    parser.add_argument('-exp',   type=str,   default='',       help='experiment differentiater string')
    parser.add_argument('-seed',  type=int,   default=None,     help='random seed of the experiment')
    parser.add_argument('-optim', type=str,   default='robust', help='optimizer to use')
    parser.add_argument('-epoch', type=int,   default=200,      help='number of epochs to run')
    args = parser.parse_args()

    identifier = args.exp
    if args.seed is not None:
        torch.manual_seed(args.seed)
        identifier += '_{}{}'.format(args.optim, args.seed)

    trainedFolder = 'Trained' + identifier
    logsFolder    = 'Logs' + identifier
    print(trainedFolder)

    os.makedirs(trainedFolder, exist_ok=True)
    os.makedirs(logsFolder   , exist_ok=True)

    with open(trainedFolder + '/args.txt', 'wt') as f:
        for arg, value in sorted(vars(args).items()):
            f.write('{} : {}\n'.format(arg, value))

    robustHandler = optim.deadZoneHandle(eta=0.5, mu=0.01, rhoBar=5, eps=2e-4)

    # Define the cuda device to run the code on.
    print('Using GPUs {}'.format(args.gpu))
    if args.cuda == -1:
        device=torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu[0]))
    # Create network instance.
    if len(args.gpu) == 1:
        net = Network(robustHandler).to(device)
        module = net
    else:
        net = torch.nn.DataParallel(Network(robustHandler).to(device), device_ids=args.gpu)
        module = net.module

    # Define optimizer module.
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr = args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optim == 'robust':
        optimizer = optim.robustDeadZone(net.parameters(), module.robustHandler, momentum=0.9, weight_decay=5e-4,)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr = args.lr, weight_decay=5e-4)
    elif args.optim == 'radam':
        optimizer = radam.RAdam(net.parameters(), lr = args.lr, weight_decay=5e-4)
    else:
        raise Exception('Optimizer {} not supported\!'.format(args.optim))

    # Dataset and dataLoader instances.
    # MNIST Dataset
    trainingSet = datasets.CIFAR10(
        root='../data/',
        train=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]),
        download=True,
    )

    testingSet = datasets.CIFAR10(
        root='../data/',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]),
    )

    trainLoader = DataLoader(dataset=trainingSet, batch_size=args.b, shuffle=True, num_workers=1)
    testLoader  = DataLoader(dataset=testingSet , batch_size=args.b, shuffle=True, num_workers=1)

    # Learning stats instance.
    stats = learningStats()

    # training loop
    milestones = [60, 120, 160]
    initialEps = robustHandler.eps
    for epoch in range(args.epoch):
        tSt = datetime.now()

        if args.optim == 'robust':
            # if epoch > milestones[0]:
            #     robustHandler.eps = initialEps * (args.epoch-epoch)/(args.epoch-milestones[0])
            # if epoch in [30, 100, 160]:
            #     for param_group in optimizer.param_groups:
            #         print('Rho increased from', robustHandler.rhoBar)
            #         stats.linesPrinted = 0
            #         robustHandler.rhoBar *= 10
            #         # robustHandler.eta *= 0.2
            pass
        else:
            if epoch in milestones:
                for param_group in optimizer.param_groups:
                    print('Learning rate reduction from', param_group['lr'])
                    stats.linesPrinted = 0
                    param_group['lr'] *= 0.2


        # Training loop.
        for i, (input, label) in enumerate(trainLoader, 0):
            net.train()

            # # Warmup for adam
            # if args.optim == 'adam' and epoch < 5:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = args.lr * (i + epoch*len(trainLoader)) / 5 *len(trainLoader) 
            # # Warmup for rhobar
            # if args.optim == 'robust' and epoch < 5:
            #     for param_group in optimizer.param_groups:
            #         robustHandler.eta = 0.5 * (1 + np.exp(-(i*epoch/5/len(trainLoader) + 1))) 

            input  = input.to(device)
            output = net.forward(input)
            output.retain_grad()
            
            prediction = output.data.max(1, keepdim=True)[1].cpu().flatten()
            stats.training.correctSamples += torch.sum( prediction == label ).data.item()
            stats.training.numSamples     += len(label)
            
            loss = F.cross_entropy(output, label.to(device))
            optimizer.zero_grad()
            loss.backward()
            if args.optim == 'robust':
                optimizer.step(error=output.grad)
            else:
                optimizer.step()

            stats.training.lossSum += loss.cpu().data.item()

            lr = []
            if args.optim == 'robust':
                for p in module.parameters():
                    if hasattr(p, 'rn'):
                        lr.append('Layer :{:3d}, lr : {:.6e}, R : {}'.format(p.rn.layer, p.rn.learningRate, p.rn.rNorm))

            # Display training stats.
            stats.print(
                epoch, i, 
                (datetime.now() - tSt).total_seconds() / (i+1) / trainLoader.batch_size,
                header = lr,
            )

            
        # Testing loop.
        for i, (input, label) in enumerate(testLoader, 0):
            net.eval()

            with torch.no_grad():
                input  = input.to(device)
                
                output = net.forward(input)

                prediction = output.data.max(1, keepdim=True)[1].cpu().flatten()
                stats.testing.correctSamples += torch.sum( prediction == label ).data.item()
                stats.testing.numSamples     += len(label)

                loss = F.cross_entropy(output, label.to(device))
                stats.testing.lossSum += loss.cpu().data.item()

            lr = []
            if args.optim == 'robust':
                for p in module.parameters():
                    if hasattr(p, 'rn'):
                        lr.append('Layer :{:3d}, lr : {:.6e}'.format(p.rn.layer, p.rn.learningRate))

            # Display training stats.
            stats.print(
                epoch, i, 
                header = lr,
            )
        
        # scheduler.step(stats.testing.accuracy())
        
        # Update stats.
        stats.update()
        stats.plot(saveFig=True, path= trainedFolder + '/')
        if stats.testing.bestAccuracy is True:  torch.save(module.state_dict(), trainedFolder + '/network.pt')

        if epoch%100 == 0:
            torch.save(
                {
                    'net': module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, 
                logsFolder + '/checkpoint%d.pt'%(epoch)
            )
            
        stats.save(trainedFolder + '/')
