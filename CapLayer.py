"""
Created on Tue May 15 02:19:37 2018

@author: rinzler
"""

from Capsule import Capsule
import torch.nn.functional as F
import torch.nn as nn

class CapLayer(nn.Module):
    def __init__(self, num_caps, in_dim, cap_dim, gen_dim):
        super(CapLayer, self).__init__()
        self.caps = nn.ModuleList([
                Capsule(in_dim, cap_dim, gen_dim)
                for _ in range(num_caps)])
#        print(len(self.caps))    
    def forward(self, X, delxy, sep = False):
        caps_out = [cap(X, delxy) for cap in self.caps]
        R = []
        for cap in self.caps:
            if not sep:
                x = cap(X,delxy)
                caps_out.append(x)
            else:
                x, y = cap(X, delxy, sep)
                caps_out.append(x)
                R.append(y)
        t = caps_out[0]
        if sep:
            r = R[0]
            for i in range(1, len(self.caps)):
                r =  (R[i] + r )/2
#        print(t.size())
        for i in range(1, len(self.caps)):
            
            t += caps_out[i]

        if not sep:
            return F.sigmoid(t)
        return F.sigmoid(t), r