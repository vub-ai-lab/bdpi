# This file is part of Bootstrapped Dual Policy Iteration
# 
# Copyright 2018, Vrije Universiteit Brussel (http://vub.ac.be)
#     authored by Denis Steckelmacher <dsteckel@ai.vub.ac.be>
#
# BDPI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BDPI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BDPI.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import lzo
import pickle
import random
import argparse

import gym
import numpy as np
import datetime

import gym_envs
from bdpi import BDPI

class Learner(object):
    def __init__(self, args):
        """ Construct a Learner from parsed arguments
        """
        self.total_timesteps = 0
        self.total_episodes = 0
        self._datetime = datetime.datetime.now()

        self._render = args.render
        self._learn_loops = args.loops
        self._learn_freq = args.erfreq
        self._offpolicy_noise = args.offpolicy_noise
        self._temp = float(args.temp)
        self._retro = args.retro

        # Make environment
        if args.retro:
            import retro

            self._env = retro.make(game=args.env)
        else:
            self._env = gym.make(args.env)

        # Observations
        self._discrete_obs = isinstance(self._env.observation_space, gym.spaces.Discrete)

        if self._discrete_obs:
            self._state_vars = self._env.observation_space.n                    # Prepare for one-hot encoding
        else:
            self._state_vars = int(np.product(self._env.observation_space.shape))

        # Primitive actions
        aspace = self._env.action_space

        if isinstance(aspace, gym.spaces.Tuple):
            aspace = aspace.spaces
        else:
            aspace = [aspace]               # Ensure that the action space is a list for all the environments

        if isinstance(aspace[0], gym.spaces.Discrete):
            # Discrete actions
            self._num_actions = int(np.prod([a.n for a in aspace]))
        elif isinstance(aspace[0], gym.spaces.MultiBinary):
            # Retro actions are binary vectors of pressed buttons. Quick HACK,
            # only press one button at a time
            self._num_actions = int(np.prod([a.n for a in aspace]))
        else:
            # Continuous actions
            print(aspace)
            raise NotImplementedError('Continuous actions are not supported')

        self._aspace = aspace

        # BDPI algorithm instance
        self._bdpi = BDPI(self._state_vars, self._num_actions, args, None)

        # Summary
        print('Number of primitive actions:', self._num_actions)
        print('Number of state variables', self._state_vars)

    def loadstore(self, filename, load=True):
        """ Load or store weights from/to a file
        """
        self._bdpi.loadstore(filename, load)

    def encode_state(self, state):
        """ Encode a raw state from Gym to a Numpy vector
        """
        if self._discrete_obs:
            # One-hot encode discrete variables
            rs = np.zeros(shape=(self._state_vars,), dtype=np.float32)
            rs[state] = 1.0
        elif isinstance(state, np.ndarray):
            rs = state.flatten().astype(np.float32)
        else:
            rs = np.array(state, dtype=np.float32)

        return rs

    def reset(self, last_reward):
        self._last_experience = None
        self._first_experience = None
        self._bdpi.reset(last_reward)

        self.total_episodes += 1

    def save_episode(self, name):
        states = []
        actions = []
        rewards = []
        entropies = []

        e = self._first_experience

        while e:
            states.append(e.state())
            actions.append(e.action)
            rewards.append(e.reward)
            entropies.append(e.entropy)

            e = e.next_experience

        s = pickle.dumps((states, actions, rewards, entropies))
        s = lzo.compress(s)
        f = open(name + '.episode', 'wb')
        f.write(s)
        f.close()

    def execute(self, env_state):
        """ Execute one episode in the environment.
        """

        done = False
        cumulative_reward = 0.0
        seen_reward = 0.0
        i = 0

        while (not done) and (i < 108000):
            # Select an action based on the current state
            self.total_timesteps += 1

            old_env_state = env_state
            state = self.encode_state(env_state)

            action, experience = self._bdpi.select_action(state, env_state)

            # Change the action if off-policy noise is to be used
            if self._offpolicy_noise and random.random() < self._temp:
                action = random.randrange(self._num_actions)
                experience.action = action

            # Manage the experience chain
            if self._first_experience is None:
                self._first_experience = experience
            if self._last_experience is not None:
                self._last_experience.next_experience = experience

            self._last_experience = experience

            # Execute the action
            if len(self._aspace) > 1:
                # Choose each of the factored action depending on the composite action
                actions = [0] * len(self._aspace)

                for j in range(len(actions)):
                    actions[j] = action % self._aspace[j].n
                    action //= self._aspace[j].n

                env_state, reward, done, __ = self._env.step(actions)
            else:
                # Simple scalar action
                if self._retro:
                    # Binary action
                    a = np.zeros((self._num_actions,), dtype=np.int8)
                    a[action] = 1
                    action = a

                env_state, reward, done, __ = self._env.step(action)

            i += 1
            public_reward = reward

            # Render the environment if needed
            if self._render > 0 and self.total_episodes >= self._render:
                self._env.render()

            # Add the reward of the action
            experience.reward = reward
            cumulative_reward += public_reward
            seen_reward += experience.reward

            # Learn from the experience buffer
            if self._learn_freq == 0:
                do_learn = done
            else:
                do_learn = (self.total_timesteps % self._learn_freq == 0)

            if do_learn:
                s = datetime.datetime.now()
                d = (s - self._datetime).total_seconds()
                print('Start Learning, in-between is %.3f seconds...' % d)

                count = self._bdpi.train()
                ns = datetime.datetime.now()
                d = (ns - s).total_seconds()
                print('Learned %i steps in %.3f seconds, %.2f timesteps per second' % (count, d, count / d))
                print('S', count / d, file=sys.stderr)
                sys.stderr.flush()
                sys.stdout.flush()
                self._datetime = ns

        return (env_state, cumulative_reward, seen_reward, done, i)

def main():
    # Parse parameters
    parser = argparse.ArgumentParser(description="Reinforcement Learning for the Gym")

    parser.add_argument("--render", type=int, default=0, help="Enable a graphical rendering of the environment after N episodes")
    parser.add_argument("--monitor", action="store_true", default=False, help="Enable Gym monitoring for this run")
    parser.add_argument("--env", required=True, type=str, help="Gym environment to use")
    parser.add_argument("--retro", action='store_true', default=False, help="The environment is a OpenAI Retro environment (not a Gym one)")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes to run")
    parser.add_argument("--name", type=str, default='', help="Experiment name")

    parser.add_argument("--erpoolsize", type=int, default=2000, help="Number of experiences stored by each option for experience replay")
    parser.add_argument("--er", type=int, default=50, help="Number of experiences used to build a replay minibatch")
    parser.add_argument("--erfreq", type=int, default=1, help="Learn using a batch of experiences every N time-steps, 0 for every episode")
    parser.add_argument("--loops", type=int, default=1, help="Number of replay batches replayed at each time-step")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs used to fit the actors and critics")

    parser.add_argument("--hidden", default=100, type=int, help="Hidden neurons of the policy network")
    parser.add_argument("--layers", default=1, type=int, help="Number of hidden layers in the networks")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate of the neural network")
    parser.add_argument("--load", type=str, help="File from which to load the neural network weights")
    parser.add_argument("--save", type=str, help="Basename of saved weight files. If not given, nothing is saved")

    parser.add_argument("--offpolicy-noise", action="store_true", default=False, help="Add some off-policy noise on the actions executed by the agent, using e-Greedy with --temp.")
    parser.add_argument("--pursuit-variant", type=str, choices=['generalized', 'ri', 'rp', 'pg'], default='rp', help="Pursuit Learning algorithm used")
    parser.add_argument("--learning-algo", type=str, choices=['egreedy', 'softmax', 'pursuit'], default='pursuit', help="Action selection method")
    parser.add_argument("--temp", type=str, default='0.1', help="Epsilon or temperature. Can be a value_factor format where value is multiplied by factor after every episode")
    parser.add_argument("--actor-count", type=int, default=1, help="Amount of 'actors' in the mixture of experts")
    parser.add_argument("--q-loops", type=int, default=10, help="Number of training iterations performed on the critic for each training epoch")

    args = parser.parse_args()

    # Instantiate learner
    learner = Learner(args)

    # Load weights if needed
    if args.load is not None:
        print('Loading', args.load)
        learner.loadstore(args.load, load=True)

    # Execute the environment and learn from it
    f = open('out-' + args.name, 'w')
    start_dt = datetime.datetime.now()

    if args.monitor:
        learner._env.monitor.start('/tmp/monitor', force=True)

    try:
        old_dt = start_dt
        avg = 0.0
        last_reward = -1e10
        reward = None

        for i in range(args.episodes):
            learner.reset(reward)

            s = learner._env.reset()
            _, reward, seen_reward, done, length = learner.execute(s)

            # Ignore perturbed episodes
            if learner._offpolicy_noise:
                learner._offpolicy_noise = False
                learner._learn_freq = 1e6
                continue

            if args.offpolicy_noise:
                learner._offpolicy_noise = True
                learner._learn_freq = args.erfreq

            # Keep track of best episodes
            if reward > last_reward:
                last_reward = reward

                learner.save_episode(args.name)

            # Average return
            if i == 0:
                avg = reward
            else:
                avg = 0.99 * avg + 0.01 * reward

            if (datetime.datetime.now() - old_dt).total_seconds() > 60.0:
                # Save weights every minute
                if args.save is not None:
                    learner.loadstore("%s-%s" % (args.save, args.name), load=False)

                # Save last episode
                learner.save_episode(args.name + '-latest')

                old_dt = datetime.datetime.now()

            print(reward, seen_reward, avg, learner.total_timesteps, (datetime.datetime.now() - start_dt).total_seconds(), length, file=f)
            print(args.name, "Cumulative reward:", reward, "; average reward:", avg, "; length:", length)
            f.flush()
    except KeyboardInterrupt:
        pass

    if args.monitor:
        learner._env.monitor.close()

    f.close()

    # Print timing statistics
    delta = datetime.datetime.now() - start_dt

    print('Learned during', str(delta).split('.')[0])
    print('Learning rate:', learner.total_timesteps / delta.total_seconds(), 'timesteps per second')

if __name__ == '__main__':
    main()
