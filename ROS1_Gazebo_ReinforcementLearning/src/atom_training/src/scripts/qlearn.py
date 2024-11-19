import random
import json
import ast

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        print("Initializing QLearn object \n")
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    #Get the current Q value associated with the action in the current state. This represents
    #the total expected rewards an agent can obtain by taking this action in the given state 
    #and following the best policy thereafter
    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    #Update the Q value in the dictionary for the state-action pair using the Q-value formula
    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    #Determines which action to take in the current state
    def chooseAction(self, state, return_q=False):
        #Iterate through the available actions and get the one with the max Q value
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            #Adding random values to encourage exploration
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))] 
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there are several state-action max values 
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        #Return the selected action
        action = self.actions[i]        
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

    def save_state(self):
        print("Saving model parameters")
        filename = '/home/abey/Desktop/Repos/Data_Analysis_Scripts/KSU_GradSchool/MTRE6800_ResearchProject/my_model.json'

        q_converted = {str(key): value for key, value in self.q.items()}
        state = {
            'q': q_converted,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'gamma': self.gamma
        }
        with open(filename, 'w') as f:
            json.dump(state, f, indent=4)
    
    def loadModelState(self):
        filename = '/home/abey/Desktop/Repos/Data_Analysis_Scripts/KSU_GradSchool/MTRE6800_ResearchProject/SavedNNParameters/Qlearn/my_model.json'

        with open(filename, 'r') as f:
            state = json.load(f)
            # Convert string keys back to tuples
            # q_converted_back = {tuple(json.loads(key)): value for key, value in state['q'].items()}
            q_converted_back = {ast.literal_eval(key): value for key, value in state['q'].items()}
            
            self.q = q_converted_back
            self.alpha = 0.4
            self.gamma = 0.85
            self.epsilon = 1.0
        
        print("Loaded previous model state")