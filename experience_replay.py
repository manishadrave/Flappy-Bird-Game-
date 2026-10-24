from collections import deque 
import random 

class ReplayMemory():
    #to create fifo - experince replay 
    def __init__(self,maxlen, seed = None):
        self.memory = deque([], maxlen= maxlen)

    #to add features in replay memory  - add, check len, extract samples  
    def append(self, new_exp):
        self.memory.append(new_exp)

    def sample(self,sample_size):
        return random.sample(self.memory, sample_size)

#to cal current buffer size 
    def __len__(self):
        return len(self.memory)

#Experience replay is a technique in deep reinforcement learning 
# where an agent stores its past interactions in a 
# memory dataset (called a replay buffer) and randomly 
# samples batches from it to train the neural network

    


