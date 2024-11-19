import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
import json 
from keras.regularizers import l2
from keras import backend as K
from keras.models import clone_model  # Import clone_model

class DDQN:
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

        # Init target model with the same architecture and weights
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())


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
    
    def update_target_model(self):
        print("Updating target model weights")
        self.target_model.set_weights(self.model.get_weights())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([self.preprocess_state(x[0])[0] for x in minibatch])
        next_states = np.array([self.preprocess_state(x[3])[0] for x in minibatch])

        # Predict Q-values for starting and next states
        q_values = self.model.predict(states)
        next_q_values = self.model.predict(next_states)
        next_q_values_target = self.target_model.predict(next_states)

        for i, (_, action, reward, _, done) in enumerate(minibatch):
            if done:
                q_values[i][action] = reward
            else:
                # Select the action for the next state using the main model
                best_action = np.argmax(next_q_values[i])
                # Evaluate the chosen action with the target model
                q_values[i][action] = reward + self.gamma * next_q_values_target[i][best_action]

        # Train the main model
        self.model.fit(states, q_values, batch_size=batch_size, verbose=0)

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

         # Load other training parameters
        self.epsilon = 1.0
        self.gamma = 1.0            
        self.alpha = 0.0075