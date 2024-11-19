import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
import json 
from keras.regularizers import l2
from keras import backend as K

class N1try:
    def __init__(self, gamma=1.0, epsilon=1.0, alpha=0.0075, l2_strength=0.01):
        self.current_episode = 0  # Initialize episode counter
        self.memory = deque(maxlen=100000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha

        self.state_size = 4
        self.action_size = 3
        self.batch_size = 64
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        
        # Init model
        self.model = Sequential([
            Dense(units=24, input_dim=self.state_size, activation='relu'),
            Dense(units=48, activation='relu', kernel_regularizer=l2(l2_strength)),
            Dense(3, activation='linear')
        ])
        self.optimizer = Adam(learning_rate=self.alpha)
        self.model.compile(loss='mse', optimizer=self.optimizer)

    def update_learning_rate(self):
        decay_rate = 0.95  # Example decay rate
        min_lr = 1e-5  # Minimum learning rate
        # Apply decay
        self.alpha *= decay_rate
        # Ensure learning rate does not fall below minimum
        self.alpha = max(self.alpha, min_lr)
        # Update the optimizer's learning rate
        K.set_value(self.model.optimizer.learning_rate, self.alpha)
        print("Updating learning rate to:", self.alpha)

    def learn(self, state, action, reward, next_state, done):
        print("Current learning rate:", self.alpha)
        self.memory.append((state, action, reward, next_state, done))

    def chooseAction(self, state, epsilon, action_space):
        preprocessed_state = self.preprocess_state(state)  # Preprocess the state
        return action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.model.predict(preprocessed_state))

    def preprocess_state(self, state):
        return np.reshape(state, [1, 4])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        for state, action, reward, next_state, done in minibatch:
            state = self.preprocess_state(state)  # Ensure state is preprocessed
            next_state = self.preprocess_state(next_state)  # Ensure next_state is preprocessed
            
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])  
            y_batch.append(y_target[0])
        
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def saveModelState(self):
        print("Saving model parameters")
        self.model.save('/home/abey/Desktop/Repos/Data_Analysis_Scripts/KSU_GradSchool/MTRE6800_ResearchProject/my_model')  # Saves to a directory 'my_model'

        # Save other training parameters
        params = {
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'alpha': self.alpha
        }
        with open(f'/home/abey/Desktop/Repos/Data_Analysis_Scripts/KSU_GradSchool/MTRE6800_ResearchProject/training_params.json', 'w') as f:
            json.dump(params, f)

    def loadModelState(self, directory):
        print("Loading previous model parameters")
        self.model = load_model(f'{directory}/my_model')

        self.epsilon = 1.0
        self.gamma = 1.0            
        self.alpha = 0.0075