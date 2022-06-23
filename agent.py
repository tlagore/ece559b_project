from collections import deque
from dataclasses import dataclass
from os import path
import random
import time
import pickle

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, losses

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
    target_network: bool = False

    # Agent config
    learning_batch_size: int = 32
    update_after_actions: int = 4
    update_target_network: int = 400
    replay_buffer_size: int = 10000
    num_episodes: int = 400
    max_steps: int = 200

@dataclass
class TestCase():
    config: AgentConfiguration
    training_rewards: dict
    test_rewards: dict
    method: StateAttributeType
    description: str

    file_format = "results/{0}.pickle"
    
    def write_test(self):
        name = ''

        if self.method == StateAttributeType.STATE_FEATURES:
            name += 'state-features'
        else:
            name += 'grid-features'

        if self.config.target_network:
            name += '_target-network'
        else:
            name += '_single-network'

        file_name = name

        i = 1
        while path.exists(self.file_format.format(file_name)):
            file_name = name + str(i)
            i += 1

        with open(self.file_format.format(file_name), 'wb') as out_file:
            pickle.dump(self, out_file)

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
        self.config = config

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
        
        self.learning_batch_size = config.learning_batch_size

        # use deque for performance over list
        self.replay_buffer = deque(maxlen = config.replay_buffer_size)
        self.game_over_buffer = deque(maxlen = config.replay_buffer_size)
        self.num_episodes = config.num_episodes
        self.max_steps = config.max_steps

        self.model = self.initialize_dqn(method)

        if self.config.target_network:
            self.target_model = self.initialize_dqn(method)

    def initialize_dqn(self, method: StateAttributeType):
        if method == StateAttributeType.STATE_FEATURES:
            return self.initialize_linear_state_dqn()

        elif method == StateAttributeType.GRID_FEATURES:
            ## attempting to use a convolution DQN using the RGB values of individual grid cells as the features
            return self.initialize_convolution_state_dqn()

    def initialize_convolution_state_dqn(self):
        ## this is not currently working, need to look into this more
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
        optimizer = keras.optimizers.Adam(learning_rate=self.alpha)
        dqn.compile(optimizer=optimizer, loss=self.loss_fn)
        return dqn

    def get_action(self, state):
        """
        """
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
            print("TRYING A RANDOM ACTION!!!")
        else:
            predicted_values = self.model(state, training=False)
            action = np.argmax(predicted_values)

        return action

    def buffer_experience(self, experience : Experience):
        self.replay_buffer.appendleft(experience)

    def replay_experience(self):
        # weight our experience with negative reward
        # game_over_experience: list[Experience] = random.choices(self.game_over_buffer, k=int(self.learning_batch_size*.2))
        batch: list[Experience] = random.choices(self.replay_buffer, k=int(self.learning_batch_size)) # self.rng.choice(self.replay_buffer, self.learning_batch_size)
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

        if self.config.target_network:
            value_predictions = rewards + self.discount*(np.amax(self.target_model.predict_on_batch(next_states), axis=1))
        else:
            value_predictions = rewards + self.discount*(np.amax(self.model.predict_on_batch(next_states), axis=1))

        values = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.learning_batch_size)])
        values[[ind], [actions]] = value_predictions

        self.model.fit(states, values, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_floor:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_floor)

    def train(self):
        """
        Train the agent
        """

        num_moves = 0

        scores = []
        rewards = []
        episode_times = []
        self.environment._render = False
        for i in range(self.num_episodes):
            self.environment.episode = i
            self.environment.reset()
            state = self.environment.get_state_features()

            # with np.printoptions(edgeitems=50):
            #     print(state)
            episode_reward = 0
            start_time = time.time()
            print(f'episode: {i}')
            for _ in range(self.max_steps):
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action = self.get_action(state_tensor)
         
                next_state, reward, done = self.environment.step(action)

                self.buffer_experience(Experience(state, action, reward, next_state, done))

                episode_reward += reward

                if self.config.target_network and num_moves % self.update_target_netork == 0:
                    self.target_model.set_weights(self.model.get_weights())

                if num_moves % self.update_after_actions == 0 and len(self.replay_buffer) >= self.learning_batch_size:
                    self.replay_experience()

                if done:
                    break

                state = next_state
                num_moves += 1

            scores.append(snake_env.last_score)
            rewards.append(episode_reward)
            episode_times.append(time.time() - start_time)

        # do a final copy of the weights to target network
        if self.config.target_network:
            self.target_model.set_weights(self.model.get_weights())


        reward_iterations = {
            'episodes': list(range(0, self.num_episodes)),
            'times': episode_times,
            'rewards': rewards,
            'scores': scores
        }

        # test the trained model on 10 episodes with no random start state
        self.environment.randomize_state = False
        self.environment._render = True
        self.epsilon = 0.0
        self.epsilon_floor = 0.0
        self.epsilon_decay = 0.0

        input("Press any key to begin testing")

        test_scores = []
        test_rewards = []
        for i in range(0, 11):
            self.environment.episode = f'{i}-test'
            self.environment.reset()
            
            done = False
            ep_reward = 0

            state = self.environment.get_state_features()
            while not done:
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action = self.get_action(state_tensor)
                next_state, reward, done = self.environment.step(action)
                ep_reward += reward
                state = next_state

            test_rewards.append(ep_reward)
            test_scores.append(self.environment.last_score)


        test_iterations = {
            'episodes': list(range(0, 11)),
            'scores': test_scores,
            'rewards': test_rewards
        }

        print(f'exiting after {self.num_episodes} episodes')

        return reward_iterations, test_iterations


if __name__ == "__main__":
    conf = SnakeConfig()
    conf.difficulty = 'easy'
    conf.grid_cell_size = 20
    conf.grid_size = 30
    conf.is_human = False
    conf.render = False
    conf.randomize_state = True
    conf.debug = False
    conf.method = StateAttributeType.STATE_FEATURES

    # state features no target network
    snake_env = SnakeEnvironment(conf)
    
    # most effective training parameters so far
    agent_config = AgentConfiguration()
    agent_config.num_episodes = 400
    agent_config.target_network = True
    agent_config.num_hidden_layers = 1
    agent_config.epsilon = 0.9
    agent_config.epsilon_decay = 0.999
    agent_config.epsilon_floor = 0.1
    alpha = 0.01
    agent = Agent(agent_config, snake_env, conf.method)
    training_scores, test_scores = agent.train()
    # test = TestCase(agent_config, training_scores, test_scores, conf.method, 'Higher batch memory, higher epsilon, lower batch size')
    # test.write_test()
    snake_env.full_reset()