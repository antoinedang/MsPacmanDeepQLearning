from agent import *
import warnings
from utils import *
import random


ideal_agent = makeAgent()

for layer in ideal_agent.model:
        if isinstance(layer, nn.Linear):
            with torch.no_grad():
                layer.weight.fill_(1.0)
                layer.bias.fill_(0.0)
                
saveToPickle("best_agent.pkl", ideal_agent)