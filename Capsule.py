import torch.nn as nn
import torch.nn.functional as F
import torch
'Defines one capsule'

class Capsule(nn.Module):
    def __init__(self, input_dim, cap_dim, gen_dim, xy_dim = 2):
        super(Capsule, self).__init__()
        self.indim = input_dim
        self.cpdim = cap_dim
        self.gndim = gen_dim
        self.xytrn = xy_dim
        self.cp = nn.Linear(self.indim, self.cpdim) #Recognizer units
        self.xy = nn.Linear(self.cpdim, self.xytrn) #estimates of the X and Y
        self.pr = nn.Linear(self.cpdim, 1)          #prob of feature
        self.gn = nn.Linear(self.xytrn, self.gndim) #The generator
        self.rc = nn.Linear(self.gndim, self.indim) #The reconstructed image

    def forward(self, X, delxy, sp = False): 
        X = X.view(-1, 28*28)
#        print(X.size(), delxy.size())
        cap = F.sigmoid(self.cp(X))
#        print('cap', cap.size())
        x_y = self.xy(cap)
        # print('x_y', x_y.size())
        prb = self.pr(cap)
#        print('prb', prb.size())
#        print('x_y + del', (x_y + delxy).size())
        gen = self.gn(x_y + delxy)
#        print('gen', gen.size())
        rec = self.rc(gen)
#        print('rec',rec.size())
#        rec = rec.view(64, 1, 28, 28)
#        torch.matmul(rec,prb)
        if sp:
            return torch.mul(rec,prb), x_y
        else:
            return torch.mul(rec,prb)
    