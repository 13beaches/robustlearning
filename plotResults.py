import sys, os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas

experiments = {
    'robust' : 'Adaptive Dead Zone',
    'sgd'    : 'SGD',
    'adam'   : 'Adam',
    'radam'  : 'RAdam'
}

keys = [
    'sgd',   
    'adam',  
    'radam', 
    'robust',
]

if __name__ == '__main__':
    trAccFig  = plt.figure(1)
    teAccFig  = plt.figure(2)
    trLossFig = plt.figure(3)
    teLossFig = plt.figure(4)
    trAcc  = trAccFig .add_subplot(1, 1, 1)
    teAcc  = teAccFig .add_subplot(1, 1, 1)
    trLoss = trLossFig.add_subplot(1, 1, 1)
    teLoss = teLossFig.add_subplot(1, 1, 1)

    for key in keys:
        exp = experiments[key]
        dirs = glob.glob('Trained_' + key + '*')

        acc = []
        loss = []
        bestAcc = []
        for dir in dirs:
            acc.append(np.loadtxt('{}/accuracy.txt'.format(dir)))
            loss.append(np.loadtxt('{}/loss.txt'.format(dir)))
            bestAcc.append([acc[-1][:,0].max(), acc[-1][:,1].max()])

        acc = np.array(acc)
        loss = np.array(loss)

        accMean = np.mean(acc, axis=0)
        lossMean = np.mean(loss, axis=0)

        bestAccMean = np.mean(bestAcc, axis=0) * 100
        bestAccStd  = np.std (bestAcc, axis=0) * 100
        print(exp)
        print('Training Accuracy: {:.2f} \pm {:.2f}'.format(bestAccMean[0], bestAccStd[0]))
        print('Testing  Accuracy: {:.2f} \pm {:.2f}'.format(bestAccMean[1], bestAccStd[1]))

        epoch = np.arange(accMean.shape[0])
        trAcc .plot(epoch,  accMean[:, 0], label=exp)
        teAcc .plot(epoch,  accMean[:, 1], label=exp)
        trLoss.semilogy(epoch, lossMean[:, 0], label=exp)
        teLoss.semilogy(epoch, lossMean[:, 1], label=exp)

        trAcc .fill_between(epoch, acc [:, :, 0].min(axis=0), acc [:, :, 0].max(axis=0), alpha=0.2)
        teAcc .fill_between(epoch, acc [:, :, 1].min(axis=0), acc [:, :, 1].max(axis=0), alpha=0.2)
        trLoss.fill_between(epoch, loss[:, :, 0].min(axis=0), loss[:, :, 0].max(axis=0), alpha=0.2)
        teLoss.fill_between(epoch, loss[:, :, 1].min(axis=0), loss[:, :, 1].max(axis=0), alpha=0.2)

    trAcc .set(xlabel='Epoch', ylabel='Training Accuracy')
    teAcc .set(xlabel='Epoch', ylabel='Testing Accuracy')
    trLoss.set(xlabel='Epoch', ylabel='Training Loss')
    teLoss.set(xlabel='Epoch', ylabel='Testing Loss')

    trAcc .legend(loc='lower right')
    teAcc .legend(loc='lower right')
    trLoss.legend(loc='upper right')
    teLoss.legend(loc='upper right')

    # plt.show()
    import tikzplotlib
    tikzplotlib.save(filepath='trainingAccuracy.tex', figure=trAccFig)
    tikzplotlib.save(filepath='testingAccuracy.tex' , figure=teAccFig)
    tikzplotlib.save(filepath='trainingLoss.tex', figure=trLossFig)
    tikzplotlib.save(filepath='testingLoss.tex' , figure=teLossFig)

