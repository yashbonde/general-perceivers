# Example from https://gym.openai.com/evaluations/eval_OeUSZwUcR2qSAqMmOE1UIw/

import numpy as np
import random
from collections import deque

import gym
from gym import wrappers
import torch
import torch.nn as nn

from gperc.utils import set_seed
from gperc import PerceiverConfig, Perceiver

import matplotlib.pyplot as plt

ACTIONS_DIM = 2
OBSERVATIONS_DIM = 4
MAX_ITERATIONS = 10 ** 6
LEARNING_RATE = 0.001

NUM_EPOCHS = 50

GAMMA = 0.99
REPLAY_MEMORY_SIZE = 1000
NUM_EPISODES = 10000
TARGET_UPDATE_FREQ = 100
MINIBATCH_SIZE = 32

RANDOM_ACTION_DECAY = 0.99
INITIAL_RANDOM_ACTION = 1

xvalues = []
yvalues = []

config = PerceiverConfig(
    input_len=1,
    input_dim=4,
    latent_len=16,
    latent_dim=16,
    output_len=1,
    output_dim=2,
    decoder_cross_attention=True,
    decoder_projection=False,
)
print(config)


class CartpoleSolver:
    def __init__(self, max_size):
        self.max_size = max_size
        self.transitions = deque()
        self.model = Perceiver(config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

    def add(self, observation, action, reward, observation2):
        if len(self.transitions) > self.max_size:
            self.transitions.popleft()
        self.transitions.append((observation, action, reward, observation2))

    def sample(self, count):
        return random.sample(self.transitions, count)

    def size(self):
        return len(self.transitions)

    def get_q(self, observation):
        np_obs = np.reshape(observation, [-1, 1, OBSERVATIONS_DIM])
        return self.model(torch.from_numpy(np_obs))

    def train(self, observations, targets):
        obs_tensor = torch.reshape(torch.Tensor(observations), (-1, 1, OBSERVATIONS_DIM))
        targets_tensor = torch.reshape(torch.stack(targets), (-1, 1, ACTIONS_DIM))
        targets_pred = self.model(obs_tensor)
        loss = self.criterion(targets_pred, targets_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, observation):
        np_obs = np.reshape(observation, [-1, 1, OBSERVATIONS_DIM])
        return self.model(torch.from_numpy(np_obs))

    def update_action(self, sample_transitions):
        random.shuffle(sample_transitions)
        batch_observations = []
        batch_targets = []

        for sample_transition in sample_transitions:
            old_observation, action, reward, observation = sample_transition
            targets = torch.reshape(self.get_q(old_observation), (ACTIONS_DIM, 1))
            targets[action] = reward
            if observation is not None:
                predictions = self.predict(observation)
                new_action = torch.argmax(predictions).item()
                targets[action] += GAMMA * predictions[0, 0, new_action]

            batch_observations.append(old_observation)
            batch_targets.append(targets)
        self.train(batch_observations, batch_targets)

    def run(self):
        steps_until_reset = TARGET_UPDATE_FREQ
        random_action_probability = INITIAL_RANDOM_ACTION
        env = gym.make("CartPole-v0")
        env = wrappers.Monitor(env, "/tmp/cartpole-experiment-1", video_callable=lambda episode_id: True, force=True)
        reward_list = []
        for episode in range(1, NUM_EPISODES + 1):
            observation = env.reset()
            for iteration in range(1, MAX_ITERATIONS + 1):
                random_action_probability *= RANDOM_ACTION_DECAY
                random_action_probability = max(random_action_probability, 0.1)
                old_observation = observation

                if random.random() < random_action_probability:
                    action = random.choice(range(ACTIONS_DIM))
                else:
                    q_values = self.get_q(observation)
                    action = torch.argmax(q_values).item()

                observation, reward, done, info = env.step(action)
                if done:
                    reward_list.append(iteration)
                    print("Episode {}, iterations: {}, Average Reward: {}".format(episode, iteration, sum(reward_list[-100:]) / 100))
                    xvalues.append(episode)
                    yvalues.append(sum(reward_list[-100:]) / 100)
                    reward = -200
                    self.add(old_observation, action, reward, None)
                    break

            self.add(old_observation, action, reward, observation)

            if self.size() >= MINIBATCH_SIZE:
                sample_transitions = self.sample(MINIBATCH_SIZE)
                self.update_action(sample_transitions)
                steps_until_reset -= 1

            if sum(reward_list[-100:]) / 100 > 195:
                print("Average Reward for consecutive 100 episodes is more than 195!")
                break


set_seed(4)
agent = CartpoleSolver(REPLAY_MEMORY_SIZE)
agent.run()
plt.plot(xvalues, yvalues)
plt.show()
