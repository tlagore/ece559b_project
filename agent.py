from collections import deque
from dataclasses import dataclass
from math import exp
import time
from turtle import update

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.backend import convert_inputs_if_ragged
from tensorflow.python.ops.gen_math_ops import Exp

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
    learning_batch_size: int = 8
    update_after_actions: int = 4
    replay_buffer_size: int = 5000
    num_episodes: int = 50
    max_steps: int = 5000

@dataclass
class Experience():
    state: list
    action: int
    reward: int
    next_state: list
    done: bool

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
        self.num_actions = len(self.actions)
        self.update_after_actions = config.update_after_actions

        # learning rate
        self.alpha = config.alpha

        self.learning_batch_size = config.learning_batch_size

        # use deque for performance over list
        self.replay_buffer = deque(maxlen = config.replay_buffer_size)
        self.num_episodes = config.num_episodes
        self.max_steps = config.max_steps

        self.model = self.initialize_dqn()
        self.target_model = self.initialize_dqn()

    def initialize_dqn(self):
        input_size = self.environment.state_space
        output_size = len(self.actions)

        inputs = keras.Input(shape=(input_size,))
        optimizer = keras.optimizers.Adam(lr=self.alpha)

        hidden_size = (2/3)*input_size+output_size

        if self.num_hidden_layers > 0:
            hidden = layers.Dense(hidden_size, activation='relu')(inputs)

            for _ in range(1, self.num_hidden_layers):
                hidden = layers.Dense(hidden_size, activation="relu")(hidden)

            output = layers.Dense(output_size, activation='softmax')(hidden)
        else:
            output = layers.Dense(output_size, activation='softmax')(inputs)

        dqn = keras.Model(inputs=[inputs], outputs=[output])
        dqn.compile(loss='mse', optimizer=optimizer)
        return dqn

    def get_action(self, state):
        """
        """

        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
        else:
            predicted_values = self.model(state, training=False)
            action = np.argmax(predicted_values)

        return action

    def buffer_experience(self, experience : Experience):
        self.replay_buffer.appendleft(experience)

    def replay_experience(self):
        batch: list[Experience] = self.rng.choice(self.replay_buffer, self.learning_batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in range(len(batch)):
            states.append(batch[i].state)
            actions.append(batch[i].action)
            rewards.append(batch[i].reward)
            next_states.append(batch[i].next_state)
            dones.append(float(batch[i].done))

        dones = tf.convert_to_tensor(dones)
        states = np.array(states)
        actions = np.array(actions)

        future_rewards = self.target_model.predict(next_states)
        updated_q_values = rewards + self.discount * tf.reduce_max(future_rewards, axis=1) #*(1-dones)-dones
        q_values = self.model(states)
        self.model.fit(states, q_values, epochs=1)

        if self.epsilon > self.epsilon_floor:
            self.epsilon *= self.epsilon_decay
        # print(updated_q_values)

        # # Only calculate loss on the updated Q-Values
        # masks = tf.one_hot(actions, self.num_actions)
        # print(masks)

        # with tf.GradientTape() as tape:
        #     # train model on the states and updated Q-Values
        #     q_values = self.model.fit(states)

        #     # Apply the masks to the Q-values to get the Q-value for action taken
        #     q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)

        print(updated_q_values)

    def train(self):
        """
        Train the agent
        """

        for i in range(self.num_episodes):
            self.environment.reset()
            state = self.environment.get_state_features()
            
            episode_reward = 0

            for j in range(self.max_steps):
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action = self.get_action(state_tensor)
                
                next_state, reward, done = self.environment.step(action)

                self.buffer_experience(Experience(state, action, reward, next_state, done))

                episode_reward += reward

                state = next_state

                if j % self.update_after_actions == 0 and len(self.replay_buffer) >= self.learning_batch_size:
                    self.replay_experience()

                if done:
                    print("DONE!")
                    # print(f'final state before dying: {str(prev_state)}')
                    # print(f'episode: {e+1}/{episode}, score: {score}')
                    break

                time.sleep(0.1)



if __name__ == "__main__":
    snake_env = SnakeEnvironment(30, 20, 'easy', is_human=False, debug=True)
    config = Configuration()
    agent = Agent(config, snake_env)
    agent.train()
