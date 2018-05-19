"""
Created on Tue May 15 02:48:08 2018

@author: rinzler
"""

from CapLayer import CapLayer
from image_utils import BatchShift
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

BATCH_SIZE = 64
NUM_EPOCH = 5

transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST('/tmp', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST('/tmp', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

def show_batch(batch,title='misc',save = False):
    im = torchvision.utils.make_grid(batch, normalize = True)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    if save:
        plt.imsave('images/{0}/ep_{1:05d}_img_{2:05d}.png'.format(title,epoch+1,i+1),np.transpose(im.numpy(), (1, 2, 0)))

#dataiter = iter(trainloader)
#images, labels = dataiter.next()
#
##print('Labels: ', labels)
#print('Batch shape: ', images.size())
#show_batch(images)
    
capL = CapLayer(50, 28*28, 10, 50)
#print(capL)
crit = nn.BCELoss()
optimizer = optim.Adam(capL.parameters(), lr = 0.001)  

#e_i = []
#e_l = []
ii = 0
s_i = []
s_l = []
for epoch in range(NUM_EPOCH):
    runn_loss = 0.0
    for i, data in enumerate(trainloader,0):
        inp, _ = data
#        print(inp[0])
        target, dxy = BatchShift(inp.numpy().copy(),[-4,4])
        target = torch.from_numpy(target)
#        print(target.size())
        dxy = Variable(torch.from_numpy(dxy).float())
        inp = Variable(inp)
        target = Variable(target)
        optimizer.zero_grad()
        out = None
        R = None
        if i % 500 == 0:
            out, R = capL(inp, dxy, sep = True)
        else:
            out = capL(inp, dxy)
        out = out.view(-1, 1, 28,28)
#        print('out', out.size())
#        print('in', inp.size())
        loss = crit(out, target)
#        print('curr_out : ', out.data)
        loss.backward()
        optimizer.step()
        runn_loss += loss.data[0]
        s_i.append(ii)
        ii+=1
        s_l.append(loss.data[0])

        if i % 500 == 0:
                file = open('data/instantiate.txt', 'a')
                file.write("ep_{0:05d}_img_{1:05d}\n".format(epoch+1,i+1))
                file.write(str(R))
                file.write('\n')
                file.close()
        if i % 100 == 0:
            show_batch(target.data, title='Target', save = True)
            show_batch(inp.data, title = 'Input', save = True)
            show_batch(out.data, title = 'output', save = True)
            print('{0:05d}, {1:05d} loss : {2:6.5f}'.format(epoch+1, i+1, runn_loss / 100))
            runn_loss = 0
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(s_i, s_l)
ax.set_title('ep_{}'.format(epoch))
fig.savefig('plots/ep_{}.png'.format(epoch))



# Showing test image
data = iter(testloader)
img, _  = data.next()

targ , dxy = BatchShift(img.numpy().copy(),[-4,4])
out, R = capL(Variable(img), Variable(torch.from_numpy(dxy).float()), sep=True)
out = out.view(-1,1,28,28)
show_batch(img, title='Test_Input')
plt.show()
show_batch(torch.from_numpy(targ), title='Test_Targer')
plt.show()
show_batch(out.data, title='Test_output')
plt.show()
print(R)