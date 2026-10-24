#contains a agent primary code
import random
import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
import torch 
from experience_replay import ReplayMemory 
import itertools #to run code indefinately 
import yaml 
import torch.nn as nn
import torch.optim as optim
import os 
import argparse

if torch.backends.mps.is_available():
    device = "mps"

elif torch.cuda.is_available():
    device = "cuda"

else:
    device = "cpu"


RUNS_DIR = 'runs' #to save directory for best performing parameters from hyperparameters 
os.makedirs(RUNS_DIR , exist_ok = True)

class Agent:

    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        #agent to have access of all parms 
        self.alpha = params["alpha"] 
        self.gamma = params["gamma"] 

        self.epsilon_init = params["epsilon_init"] 
        self.epsilon_min = params["epsilon_min"] 
        self.epsilon_decay = params["epsilon_decay"] 

        self.replay_memory_size = params["replay_memory_size"] 
        self.mini_batch_size = params["mini_batch_size"] 

        self.reward_threshold = params["reward_threshold"] 
        self.network_sync_rate = params["network_sync_rate"]

        self.loss_fn = nn.MSELoss() #loss function for DQN
        self.optimizer = None 

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR,f"{self.param_set}.pt")

    def run(self, is_training = True,render = False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)
        num_states = env.observation_space.shape[0] #i/p dim
        num_actions = env.action_space.n #o/p dim

        policy_dqn = DQN(num_states,num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size) #to have dynemic value 
            epsilon = self.epsilon_init #initial exploration rate

            #to create target dqn
            target_dqn = DQN(num_states,num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0 

            self.optimizer = optim.Adam(policy_dqn.parameters(), lr = self.alpha)
            best_reward = float("-inf") #to initailize from -ve infinty 

        else:
            #to load best model AKA best policy 
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

            
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype = torch.float, device = device) #to convert into tensor 

            episode_reward = 0 
            terminated = False
            done = False 
         
            while not done and episode_reward < self.reward_threshold:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample() #explore
                    action = torch.tensor(action, dtype = torch.long, device = device)

                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim = 0)).squeeze().argmax() #exploit  
                 # Next action:
                 # (feed the observation to your agent here)
        

            # Processing: # terminated used as done variable
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                episode_reward += reward

                #to create tensor 
                reward = torch.tensor(reward,dtype = torch.float, device = device)
                next_state = torch.tensor(next_state, dtype = torch.float, device = device)
             
                if is_training: #for experience replay - store the experience in memory
                   memory.append((state,action,next_state,reward, done))
                   steps += 1

                state = next_state

            if is_training:
                print(f"Episode = {episode + 1}, Reward = {episode_reward:.2f}, Epsilon = {epsilon:.3f}")
                
            else:
                print(f"Episode = {episode + 1}, Reward = {episode_reward:.2f}")

            if is_training: #epsilon decay at end of every episode
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

#to save best reward in log file 
                if episode_reward > best_reward:
                    log_msg = f"best reward = {episode_reward} for episode = {episode + 1}" 

                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward


#to sync the target dqn with policy dqn after every network_sync_rate steps
                if len(memory) >= self.mini_batch_size: #get sample 
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch,policy_dqn,target_dqn)

                #sync the network  
                if steps > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0 

    def optimize(self,mini_batch, policy_dqn, target_dqn):
        #to get experinces - training in a batch 
        states, action, next_states, rewards,terminations = zip(*mini_batch)

        states = torch.stack(states)
        action = torch.stack(action) 
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float32, device=device)
        #cal target Q vals if termination is true
        
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(dim =1)[0]
           
        #cal y pred - q values from current policy 
        current_q = policy_dqn(states).gather(dim =1, index = action.unsqueeze(dim =1)).squeeze(dim = 1)

        #to compute loss y pred - y true 
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward() #for backprop 
        self.optimizer.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("hyperparameters", help="")
    parser.add_argument("--train", help="Training mode", action="store_true")
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)
    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)


    #env.close() - to manually stop and run indefinately 


