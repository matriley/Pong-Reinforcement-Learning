# Pong Using The Neural Network
This is a Python implementation of the classic game Pong, using reinforcement learning. The game is played by a neural network agent, which learns to play the game by trial and error. The agent receives the state of the game as input, and outputs an action (move the paddle up, down or stay in place).

# Requirements
```
Python 3.9
Pygame
Numpy
Tensorflow 2.12.0
```
# Usage
To run the game and train the agent, simply run the train.py file. The agent will play 1000 games to train and the trained model will be saved as a .h5 file.

# Usage
To use this program, cd into the folder and simply run the train.py file or run the file in a Python environment.
```
$ git clone https://github.com/matriley/Pong-Reinforcement-Learning.git
$ python3 train.py
```

# How It Works
The code is written in Python 3.9, and uses the Pygame library for game rendering, Numpy for numerical computations and Tensorflow for building the neural network.

The game is played by a neural network with three layers. The first layer has 128 neurons and uses the ReLU activation function, the second layer has 64 neurons and also uses ReLU activation, and the third layer has 3 neurons and uses linear activation. The neural network is trained using the Mean Squared Error loss function and the Adam optimizer with a learning rate of 0.001.

The game state is represented by the ball's x and y coordinates, the ball's horizontal and vertical velocity, and the y coordinate of the left paddle. The state is preprocessed by converting it to a numpy array and reshaping it to a 1D array.

The agent receives the state as input and outputs an action. The action is determined by the index of the highest output neuron in the neural network's output layer. The agent can move the paddle up, down, or stay in place.

The agent's performance is evaluated using the reward system. If the agent wins the game, it receives a reward of 1, if it loses, it receives a reward of -1, and if the game ends in a tie, it receives a reward of 0.

# Future Updates
The code is still a work in progress, and there will be further changes to reach the final goal of the project. 

* Performance Updates are the priority before moving forward. A new rendering library will be used in the future. 

