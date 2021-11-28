from collections import deque
from dataclasses import dataclass
import random
import time

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.python.keras.backend import convert_inputs_if_ragged
from tensorflow.python.ops.gen_math_ops import Exp
from tensorflow.python.ops.math_ops import multiply

import pandas as pd

from snake import SnakeEnvironment, SnakeConfig, StateAttributeType

@dataclass
class AgentConfiguration():
    # DQN config
    epsilon: float = 0.05
    epsilon_floor: float = 0.001
    epsilon_decay: float = 0.999
    discount: float = 0.95
    alpha: float = 0.01
    num_hidden_layers: int = 1

    # Agent config
    learning_batch_size: int = 100
    update_after_actions: int = 6
    update_target_network: int = 100
    replay_buffer_size: int = 800
    num_episodes: int = 1000
    max_steps: int = 4000

@dataclass
class Experience():
    state: list
    action: int
    reward: int
    next_state: list
    done: bool

class Agent():
    def __init__(self, config : AgentConfiguration, snake_environment : SnakeEnvironment, method : StateAttributeType):
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

        self.update_target_netork = config.update_target_network

        self.loss_fn = losses.MeanSquaredError()

        # learning rate
        self.alpha = config.alpha
        self.optimizer = keras.optimizers.Adam(learning_rate=self.alpha)

        self.learning_batch_size = config.learning_batch_size

        # use deque for performance over list
        self.replay_buffer = deque(maxlen = config.replay_buffer_size)
        self.num_episodes = config.num_episodes
        self.max_steps = config.max_steps

        self.model = self.initialize_dqn(method)
        # self.target_model = self.initialize_dqn(method)

    def initialize_dqn(self, method: StateAttributeType):
        if method == StateAttributeType.LINEAR:
            return self.initialize_linear_state_dqn()

        elif method == StateAttributeType.CONVOLUTION:
            return self.initialize_convolution_state_dqn()

    def initialize_convolution_state_dqn(self):
        return self.initialize_linear_state_dqn()
        """
        input_shape = self.environment.state_space_shape
        # input_size = self.environment.state_space_size
        output_size = len(self.actions)

        inputs = keras.Input(shape=input_shape)

        # hidden_size = (2/3)*input_size+output_size

        # output 8 filters, 3x3 kernel size
        convolve_1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        convolve_2 = layers.Conv2D(64, 4, strides=2, activation="relu")(convolve_1)
        convolve_2 = layers.Conv2D(64, 3, strides=1, activation="relu")(convolve_1)
        flatten_layer = layers.Flatten()(convolve_2)
        dense_layer = layers.Dense(512, activation="relu")(flatten_layer)
        output = layers.Dense(output_size, activation='softmax')(dense_layer)

        dqn = keras.Model(inputs=[inputs], outputs=[output])
        dqn.compile(optimizer=self.optimizer, loss=self.loss_fn)
        print(dqn.summary())
        return dqn
        """

    def initialize_linear_state_dqn(self):
        input_size = self.environment.state_space_size
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

        dqn = keras.Model(inputs=[inputs], outputs=[output])
        dqn.compile(optimizer=self.optimizer, loss=self.loss_fn)
        return dqn

    def get_action(self, state):
        """
        """

        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
            print("TRYING A RANDOM ACTION!!!")
        else:
            predicted_values = self.model(state, training=False)
            action = np.argmax(predicted_values)

        return action

    def buffer_experience(self, experience : Experience):
        self.replay_buffer.appendleft(experience)

    def replay_experience(self):
        batch: list[Experience] = random.choices(self.replay_buffer, k=self.learning_batch_size) # self.rng.choice(self.replay_buffer, self.learning_batch_size)
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
        next_states = np.array(next_states)
        actions = np.array(actions)

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        # targets = rewards + self.discount*(np.amax(self.target_model.predict_on_batch(next_states), axis=1))
        targets = rewards + self.discount*(np.amax(self.model.predict_on_batch(next_states), axis=1))
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.learning_batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)

        # future_rewards = self.model.predict(next_states)
        # # future_rewards = self.target_model.predict(next_states)
        # updated_q_values = rewards + self.discount * np.amax(future_rewards, axis=1)*(1-dones)-dones

        # masks = tf.one_hot(actions, self.num_actions)

        # with tf.GradientTape() as tape:
        #     # Train the model on the states and updated Q-values
        #     q_values = self.model(states)

        #     # Apply the masks to the Q-values to get the Q-value for action taken
        #     q_action = tf.reduce_sum(q_values, axis=1)
        #     # Calculate loss between new Q-value and old Q-value
        #     loss = self.loss_fn(updated_q_values, q_action)

        # # Backpropagation
        # grads = tape.gradient(loss, self.model.trainable_variables)
        # self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.epsilon > self.epsilon_floor:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_floor)

    def train(self):
        """
        Train the agent
        """

        num_moves = 0

        rewards = []
        for i in range(self.num_episodes):
            self.environment.reset()
            state = self.environment.get_state_features()

            # with np.printoptions(edgeitems=50):
            #     print(state)
            episode_reward = 0
            print(f'episode: {i}')
            for j in range(self.max_steps):
                # df = pd.DataFrame(state.reshape(33, 33))
                # df.to_csv('state_rgb.csv', index=False)
                # exit()

                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action = self.get_action(state_tensor)

                # start = time.time()
                next_state, reward, done = self.environment.step(action)
                # print(f'step took {time.time() - start}')

                self.buffer_experience(Experience(state, action, reward, next_state, done))

                episode_reward += reward

                # if num_moves % self.update_target_netork == 0:
                #     self.target_model.set_weights(self.model.get_weights())

                if num_moves % self.update_after_actions == 0 and len(self.replay_buffer) >= self.learning_batch_size:
                    self.replay_experience()

                if done:
                    # print("DONE!")
                    # print(f'snake: {self.environment.snake.xcor()},{self.environment.snake.ycor()}: {state}')
                    # time.sleep(10)
                    # print(f'episode: {e+1}/{episode}, score: {score}')
                    break

                state = next_state
                if i > 50:
                    #time.sleep(0.05)
                    self.environment._render = True
                    # time.sleep(0.1)

                if i > 100:
                    self.epsilon_floor = 0.0
                    self.epsilon = 0.0

                num_moves += 1

            rewards.append(episode_reward)

if __name__ == "__main__":
    conf = SnakeConfig()
    conf.difficulty = 'easy'
    conf.grid_cell_size = 20
    conf.grid_size = 30
    conf.is_human = False
    conf.render = False
    conf.randomize_state = True
    conf.debug = False
    conf.method = StateAttributeType.LINEAR

    snake_env = SnakeEnvironment(conf)
    agent_config = AgentConfiguration()
    agent_config.num_hidden_layers = 2
    agent = Agent(agent_config, snake_env, conf.method)
    agent.train()
