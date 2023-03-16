import gymnasium as gym
from gymnasium import spaces
from typing import Union, Optional, Tuple
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt


class StablizerOneD(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    def __init__(self, config=None, noise_level=0.05, seed=1, end_step=100):
        super(StablizerOneD, self).__init__()

        self.config = config
        self.stepCount = 0
        self.currentState = 0.0
        self.targetState = 0.0
        self.nbActions = 3
        self.stateDim = 1
        self.noise_level = noise_level
        self.endStep = end_step
        self.randomSeed = seed

        # import parameter for vector env
        self.viewer = None
        self.steps_beyond_terminated = None
        self.infoDict = {"reset": False, "endBeforeDone": False, "stepCount": 0}

        # three discrete actions, 0 no movement, 1 move to positive, 2 move to negative
        self.action_space = spaces.Discrete(self.nbActions)

        high = np.array(
            [-10.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def step_count(self):
        return self.stepCount

    def step(self, action):
        if self.stepCount == 0:
            self.infoDict["reset"] = True
        else:
            self.infoDict["reset"] = False

        self.infoDict["endBeforeDone"] = False
        self.stepCount += 1
        self.infoDict["stepCount"] = self.stepCount
        self.currentState += (random.random() - 0.5) * self.noise_level
        if action == 1:  # move to positive
            self.currentState += 0.1
        if action == 2:  # move to negative
            self.currentState -= 0.1

        reward = 0.0
        done = False
        if abs(self.currentState) < 0.1:
            reward = 1.0
            done = True
            self.infoDict["done_state"] = self.currentState

        if self.stepCount > self.endStep:
            done = True
            self.infoDict["endBeforeDone"] = True

        return np.array([self.currentState]), reward, done, False, self.infoDict.copy()

    def reset(self):
        # self.infoDict['reset'] = True
        self.infoDict["stepCount"] = 0
        self.stepCount = 0
        self.currentState = (random.random() - 0.5) * 5
        return np.array([self.currentState]), {}

    def close(self):
        pass

    def seed(self):
        pass

    def render(self, mode="human"):
        pass

    def render_traj(self, stateSet, ax):
        ax.plot(np.array(stateSet)[:])

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        # the memo dict, where id-to-object correspondence is kept to reconstruct
        # complex object graphs perfectly
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


class StablizerOneDContinuous(gym.Env):
    def __init__(self, config=None, seed=1):
        super(StablizerOneDContinuous, self).__init__()

        self.config = config
        self.stepCount = 0
        self.currentState = 0.0
        self.targetState = 0.0
        self.nbActions = 1
        self.stateDim = 1
        self.endStep = 200
        self.randomSeed = seed

        # import parameter for vector env
        self.viewer = None
        self.steps_beyond_done = None
        self.infoDict = {"reset": False, "endBeforeDone": False, "stepCount": 0}

    def step_count(self):
        return self.stepCount

    def step(self, action):
        self.stepCount += 1
        self.infoDict["stepCount"] = self.stepCount
        self.currentState[0] += (random.random() - 0.5) * 0.1

        # action is the movement amount
        self.currentState += action
        reward = 0.0
        done = False
        if abs(self.currentState) < 0.5:
            reward = 1
            done = True
            self.infoDict["done_state"] = self.currentState.copy()

        if abs(self.currentState) > 5:
            reward = -1

        return self.currentState.copy(), reward, done, self.infoDict.copy()

    def reset(self):
        # self.infoDict['reset'] = True
        self.infoDict = {}
        self.infoDict["stepCount"] = 0
        self.stepCount = 0
        self.currentState = np.array([(random.random() - 0.5) * 10], dtype=np.float)
        self.infoDict["initial state"] = self.currentState.copy()
        return self.currentState


class StablizerOneDContinuousFiniteHorizon(gym.Env):
    def __init__(self, config=None, seed=1):
        super(StablizerOneDContinuousFiniteHorizon, self).__init__()

        self.config = config
        self.stepCount = 0
        self.currentState = 0.0
        self.targetState = 0.0
        self.nbActions = 1
        self.stateDim = 1
        self.endStep = 10

        if "episodeLength" in self.config:
            self.endStep = self.config["episodeLength"]

        self.randomSeed = seed

        # import parameter for vector env
        self.viewer = None
        self.steps_beyond_done = None
        self.infoDict = {"reset": False, "endBeforeDone": False, "stepCount": 0}

    def step_count(self):
        return self.stepCount

    def step(self, action):
        self.stepCount += 1
        self.infoDict["timeStep"] = self.stepCount
        self.currentState[0] += (random.random() - 0.5) * 0.1

        # action is the movement amount
        self.currentState += action
        reward = 0.0
        done = False
        if abs(self.currentState) < 0.1:
            reward = 1
            done = True
            self.infoDict["done_state"] = self.currentState.copy()

        if abs(self.currentState) > 5:
            reward = -1

        if self.stepCount == self.endStep:
            done = True

        # if abs(self.currentState) < 5:
        #     reward = - abs(self.currentState[0])
        #     done = False
        # else:
        #     reward = - abs(self.currentState[0]) * (self.endStep - self.stepCount)
        #     done = True
        #     self.infoDict['done_state'] = self.currentState

        combinedState = {"state": self.currentState.copy(), "timeStep": self.stepCount}

        return combinedState, reward, done, self.infoDict.copy()

    def reset(self):
        # self.infoDict['reset'] = True
        self.infoDict = {}
        self.infoDict["timeStep"] = 0
        self.stepCount = 0
        self.currentState = np.array([(random.random() - 0.5) * 10], dtype=np.float)
        self.infoDict["initial state"] = self.currentState.copy()
        combinedState = {"state": self.currentState.copy(), "timeStep": self.stepCount}

        return combinedState
