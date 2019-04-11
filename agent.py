import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
itr_no = 0
model_save_path = 'models/{0}.pth'.format(itr_no)


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.in_features = 20
        self.out_features = 5
        self.linear1 = nn.Linear(self.in_features, 100)
        self.linear2 = nn.Linear(100, 40)
        self.linear3 = nn.Linear(40, self.out_features)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        #x = nn.Dropout(0.6)(x)
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size()[0])):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

model = None    

if resume:
    model = PolicyNetwork()
    model.load_state_dict(torch.load(model_save_path))
else:
    model = PolicyNetwork()











