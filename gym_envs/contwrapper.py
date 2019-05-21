import gym
from gym import spaces
import numpy as np

def cat(x, num):
    return np.concatenate((x, np.zeros((num,), dtype=np.float32)))

class ContWrapper(gym.Env):
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.num_actions = wrapped.action_space.shape[0]

        self.action_space = spaces.Discrete(self.num_actions * 2 + 1)           # Increment, decrement each action, then commit
        self.observation_space = spaces.Box(
            low=cat(self.wrapped.observation_space.low, self.num_actions),
            high=cat(self.wrapped.observation_space.high, self.num_actions)
        )

        self._action = np.zeros((self.num_actions,), dtype=np.float32)
        self._deltas = np.ones_like(self._action)

    def render(self, *args):
        return self.wrapped.render(*args)

    def reset(self):
        # Start with a zero action
        self._action.fill(0.0)
        self._deltas.fill(1.0)
        self._last_state = self.wrapped.reset()
        self._count = 0

        return self.observe()

    def step(self, action):
        if action == 0 or self._count >= self.num_actions * 4:
            # Commit the action
            self._action = np.minimum(self._action, self.wrapped.action_space.high)
            self._action = np.maximum(self._action, self.wrapped.action_space.low)

            self._last_state, r, d, info = self.wrapped.step(self._action)
            self._count = 0
            self._deltas.fill(1.0)

            return self.observe(), r, d, info
        else:
            # Move the action a bit
            var_index = (action - 1) // 2
            direction = 1.0 if ((action - 1) % 2 == 0) else -1.0

            self._action[var_index] += direction * self._deltas[var_index]
            self._deltas[var_index] *= 0.5
            self._count += 1

            return self.observe(), 0.0, False, None

    def observe(self):
        """ Make an observation from the current action and last observation from the environment
        """
        return np.concatenate((self._last_state, self._action))
