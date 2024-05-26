import random
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from utils.utils import *
from utils.config import *
import sys
# sys.path.append("C:/Users/chant/OneDrive/Desktop/thesis_code")

class DQNAgent:
    def __init__(self, state_size, action_size, player_id, tensorboard_callback, learning_rate=0.005, gamma=0.8, epsilon=1.0, epsilon_min=0.001, epsilon_decay=0.999):
        self.state_size = state_size
        self.action_size = action_size
        self.player_id = player_id  # Identifier for the player (1 or 2)
        self.memory = deque(maxlen=2000)  # Memory buffer for storing experiences
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,clipnorm=0.001)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.tensorboard_callback = tensorboard_callback

    def _build_model(self):
        """ Neural Net for Deep-Q learning Model. """

        # initializer = tf.keras.initializers.HeNormal()
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

        model = tf.keras.Sequential([
            layers.Dense(128, input_dim=self.state_size, activation='relu', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(128, activation='relu', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(self.action_size, activation='linear', kernel_initializer=initializer)
        ])

        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def dqn_act(self, state, agent, board):
        valid_actions = possible_states(agent.player_id, state, board)
        if not valid_actions:
            return None, INVALID_ACTION_PENALTY  # Return None for action and a penalty for invalid state

        if np.random.rand() <= agent.epsilon:
            return random.choice(valid_actions), 0
        else:
            act_values = agent.model.predict(state)
            act_values[0][[action for action in range(agent.action_size) if action not in valid_actions]] = float('-inf')
            return np.argmax(act_values[0]), 0

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        epsilon = 1e-8

        for state, action, reward, next_state, done in minibatch:
            state = np.squeeze(state)
            state = (state - np.min(state)) / (np.max(state) - np.min(state) + epsilon)
            next_state = np.squeeze(next_state)
            next_state = (next_state - np.min(next_state)) / (np.max(next_state) - np.min(next_state) + epsilon)

            target = reward
            if not done:
                target += self.gamma * np.amax(self.target_model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target

            history = self.model.fit(np.array([state]), target_f, epochs=1, verbose=2)
            loss = history.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss
    
    def adjust_learning_rate(self, new_lr):
        self.learning_rate = new_lr

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        model_json = self.model.to_json()
        with open(name, "w") as json_file:
            json_file.write(model_json)

    def load_model(self, name):
        json_file = open(name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        return loaded_model_json
