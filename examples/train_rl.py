# Example from https://gym.openai.com/evaluations/eval_OeUSZwUcR2qSAqMmOE1UIw/

import numpy as np
import random
from collections import deque

import gym
from gym import wrappers
import torch
import torch.nn as nn

ACTIONS_DIM = 2
OBSERVATIONS_DIM = 4
MAX_ITERATIONS = 10**6
LEARNING_RATE = 0.001

NUM_EPOCHS = 50

GAMMA = 0.99
REPLAY_MEMORY_SIZE = 1000
NUM_EPISODES = 10000
TARGET_UPDATE_FREQ = 100
MINIBATCH_SIZE = 32

RANDOM_ACTION_DECAY = 0.99
INITIAL_RANDOM_ACTION = 1

class ReplayBuffer():

  def __init__(self, max_size):
    self.max_size = max_size
    self.transitions = deque()

  def add(self, observation, action, reward, observation2):
    if len(self.transitions) > self.max_size:
      self.transitions.popleft()
    self.transitions.append((observation, action, reward, observation2))

  def sample(self, count):
    return random.sample(self.transitions, count)

  def size(self):
    return len(self.transitions)

def get_q(model, observation):
  np_obs = np.reshape(observation, [-1, OBSERVATIONS_DIM])
  return model(torch.from_numpy(np_obs))

def train(model, observations, targets):
  obs_tensor = torch.reshape(torch.Tensor(observations), (-1, OBSERVATIONS_DIM))
  targets_tensor = torch.reshape(torch.stack(targets), (-1, ACTIONS_DIM))
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
  targets_pred = model(obs_tensor)
  loss = criterion(targets_pred, targets_tensor)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

def predict(model, observation):
  np_obs = np.reshape(observation, [-1, OBSERVATIONS_DIM])
  return model(torch.from_numpy(np_obs))

def get_model():
  model = nn.Sequential(
            nn.Linear(OBSERVATIONS_DIM,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,2)
        )
  return model

def update_action(action_model, target_model, sample_transitions):
  random.shuffle(sample_transitions)
  batch_observations = []
  batch_targets = []

  for sample_transition in sample_transitions:
    old_observation, action, reward, observation = sample_transition

    targets = torch.reshape(get_q(action_model, old_observation), (ACTIONS_DIM,1))
    targets[action] = reward
    if observation is not None:
      predictions = predict(target_model, observation)
      new_action = torch.argmax(predictions).item()
      targets[action] += GAMMA * predictions[0, new_action]

    batch_observations.append(old_observation)
    batch_targets.append(targets)

  train(action_model, batch_observations, batch_targets)

def main():
  steps_until_reset = TARGET_UPDATE_FREQ
  random_action_probability = INITIAL_RANDOM_ACTION

  replay = ReplayBuffer(REPLAY_MEMORY_SIZE)

  action_model = get_model()


  env = gym.make('CartPole-v0')
  env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1',video_callable=lambda episode_id: True,force=True)

  for episode in range(NUM_EPISODES):
    observation = env.reset()

    for iteration in range(MAX_ITERATIONS):
      random_action_probability *= RANDOM_ACTION_DECAY
      random_action_probability = max(random_action_probability, 0.1)
      old_observation = observation


      if np.random.random() < random_action_probability:
        action = np.random.choice(range(ACTIONS_DIM))
      else:
        q_values = get_q(action_model, observation)
        action = torch.argmax(q_values).item()

      observation, reward, done, info = env.step(action)

      if done:
        print('Episode {}, iterations: {}'.format(
          episode,
          iteration
        ))

        reward = -200
        replay.add(old_observation, action, reward, None)
        break

      replay.add(old_observation, action, reward, observation)

      if replay.size() >= MINIBATCH_SIZE:
        sample_transitions = replay.sample(MINIBATCH_SIZE)
        update_action(action_model, action_model, sample_transitions)
        steps_until_reset -= 1


if __name__ == "__main__":
  main()
