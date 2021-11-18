from collections import deque
from dataclasses import dataclass
from math import exp
import time

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.backend import convert_inputs_if_ragged

from snake import SnakeEnvironment

@dataclass
class Configuration():
    # DQN config
    epsilon: float = 0.01
    epsilon_floor: float = 0.001
    epsilon_decay: float = 0.999
    discount: float = 0.90
    alpha: float = 0.01
    num_hidden_layers: int = 1

    # Agent config
    learning_batch_size: int = 500
    replay_buffer_size: int = 5000
    num_episodes: int = 50
    max_steps: int = 5000

@dataclass
class Experience():
    state: list
    action: int
    reward: int
    next_state: list

class Agent():
    def __init__(self, config : Configuration, snake_environment : SnakeEnvironment):
        self.environment = snake_environment

        self.rng = np.random.default_rng(int(time.time()))

        self.epsilon = config.epsilon
        self.epsilon_floor = config.epsilon_floor
        self.epsilon_decay = config.epsilon_decay
        self.discount = config.discount
        self.num_hidden_layers = config.num_hidden_layers
        self.actions = self.environment.action_space

        # learning rate
        self.alpha = config.alpha

        self.learning_batch_size = config.learning_batch_size

        # use deque for performance over list
        self.replay_buffer = deque(maxlen = config.learning_batch_size)
        self.num_episodes = config.num_episodes
        self.max_steps = config.max_steps

        self.initialize_model()

    def initialize_model(self):
        input_size = self.environment.state_space
        output_size = len(self.actions)

        inputs = keras.Input(shape=(input_size,))

        hidden_size = (2/3)*input_size+output_size

        if self.num_hidden_layers > 0:
            hidden = layers.Dense(hidden_size, activation='relu')(inputs)

            for _ in range(1, self.num_hidden_layers):
                hidden = layers.Dense(hidden_size, activation="relu")(hidden)

            output = layers.Dense(output_size, activation='softmax')(hidden)
        else:
            output = layers.Dense(output_size, activation='softmax')(inputs)

        self.dqn = keras.Model(inputs=[inputs], outputs=[output])
        self.dqn.compile(loss='mse', optimizer='sgd')
        self.dqn.summary()

    def get_action(self, state):
        """
        """

        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
        else:
            predicted_values = self.dqn.predict(state)
            action = np.argmax(predicted_values)

        return action

    def buffer_experience(self, experience : Experience):
        self.replay_buffer.appendleft(experience)

    def replay_experience(self):
        batch = np.random.sample(self.replay_buffer, self.learning_batch_size)
        

    def train(self):
        """
        Train the agent
        """

        self.environment.reset()

        for i in range(self.num_episodes):
            state = self.environment.get_state_features()

            for j in range(self.max_steps):
                action = self.get_action(state)
                next_state, reward, game_over = self.environment.step(action)


if __name__ == "__main__":
    config = Configuration()
    agent = Agent(config)
