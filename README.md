# Flappy Bird Game 

Purpose: The agent is trying to solve or win the game by crossing the pipe using Reinforcement Learning  

Library Details: Flappy Bird uses Gymnasium environment version 0 in this project, where it receives +1 or -1 based on the movement

Technologies used: Deep Q-Learning is used for forming a neural network; NN library like PyTorch is utilised; the project is divided into agent, dql, and experience replay files to change the hyperparameters easily 

To train the agent: Epsilon decay is used; target network and policy network are created : inorder to compute Q values, loss and backpropagation 

Evaluation parameters used: Accuracy Score, F1 Score, Precision, Recall & Confusion Matrix 

Steps Followed : 0. Setting up the gymnasium environment
1. Define DQN 
2. Creating Experience Replay
3. Setup: Multiple Episodes of DQN
4. Epsilon Decay: Epsilon Greedy Policy + Hyperparameters
5. Target Network Creation 
6. Train & Test the project 

Result: 
The agent in testing mode is capable of crossing 2 pipes based on 10,000 training episodes 

Future Scope: The agent can be trained on more episodes to cross many pipes in the game 
