# This file is part of Bootstrapped Dual Policy Iteration
#
# Copyright 2018-2019, Vrije Universiteit Brussel (http://vub.ac.be)
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
import threading
import multiprocessing

import gym_envs
import gym_envs.contwrapper

from bdpi import BDPI
import atariwrap

# Import additional environments if available
try:
    import gym_miniworld
except:
    pass

try:
    import roboschool
except:
    pass

class Learner(object):
    def __init__(self, args, task):
        """ Construct a Learner from parsed arguments
        """
        self.total_timesteps = 0
        self.total_episodes = 0
        self._datetime = datetime.datetime.now()

        self._async_actor = args.async_actor
        self._render = args.render
        self._learn_loops = args.loops
        self._learn_freq = args.erfreq
        self._atari = args.atari
        self._retro = args.retro
        self._offpolicy_noise = args.offpolicy_noise
        self._temp = float(args.temp.split('_')[0])
        self._task = task

        # Make environment
        if args.retro:
            import retro

            self._env = retro.make(game=args.env)
        elif args.atari:
            self._env = make_atari(args.env)
            self._env = wrap_deepmind(self._env)
        else:
            self._env = gym.make(args.env)

        if isinstance(self._env.action_space, gym.spaces.Box):
            # Wrap continuous-action environments
            self._env = gym_envs.contwrapper.ContWrapper(self._env)

        # Observations
        ob = self._env.observation_space
        self._discrete_obs = isinstance(ob, gym.spaces.Discrete)

        if self._discrete_obs:
            self._state_shape = (ob.n,)                # Prepare for one-hot encoding
        else:
            self._state_shape = ob.shape

            if len(self._state_shape) > 1:
                # Fix 2D shape for PyTorch
                s = self._state_shape

                self._state_shape = (s[2], s[0], s[1])

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

        self._aspace = aspace

        # BDPI algorithm instance
        self._bdpi = BDPI(self._state_shape, self._num_actions, args)

        # Summary
        print('Number of primitive actions:', self._num_actions)
        print('State shape:', self._state_shape)

    def loadstore(self, filename, load=True):
        """ Load or store weights from/to a file
        """
        self._bdpi.loadstore(filename, load)

    def encode_state(self, state):
        """ Encode a raw state from Gym to a Numpy vector
        """
        if self._discrete_obs:
            # One-hot encode discrete variables
            rs = np.zeros(shape=self._state_shape, dtype=np.float32)
            rs[state] = 1.0
            return rs
        elif len(state.shape) > 1:
            # Atari, retro and other image-based are NHWC, PyTorch is NCHW
            return np.swapaxes(state, 2, 0)
        else:
            return np.asarray(state, dtype=np.float32)

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
        index = self._bdpi._experiences.index(e)

        for e in list(self._bdpi._experiences)[index:]:
            states.append(e.state())
            actions.append(e.action)
            rewards.append(e.reward)
            entropies.append(e.entropy)

        with open(name + '.episode', 'wb') as f:
            f.write(lzo.compress(pickle.dumps((states, actions, rewards, entropies))))

        with open('/tmp/' + name + '-buffer.picklez', 'wb') as f:
            f.write(lzo.compress(pickle.dumps(list(self._bdpi._experiences))))

    def execute(self, env_state):
        """ Execute one episode in the environment.
        """

        done = False
        cumulative_reward = 0.0
        i = 0

        while (not done) and (i < 108000):
            # Select an action based on the current state
            self.total_timesteps += 1

            old_env_state = env_state
            state = self.encode_state(env_state)

            action, experience = self._bdpi.select_action(state)

            # Change the action if off-policy noise is to be used
            if self._offpolicy_noise and random.random() < self._temp:
                action = random.randrange(self._num_actions)
                experience.action = action

            # Manage the experience chain
            if self._first_experience is None:
                self._first_experience = experience
            if self._last_experience is not None:
                self._last_experience.set_next(experience)

            self._last_experience = experience

            # Execute the action
            if len(self._aspace) > 1:
                # Choose each of the factored action depending on the composite action
                actions = [0] * len(self._aspace)

                for j in range(len(actions)):
                    actions[j] = action % self._aspace[j].n
                    action //= self._aspace[j].n

                env_state, reward, done, _ = self._env.step(actions)
            else:
                # Simple scalar action
                if self._retro:
                    # Binary action
                    a = np.zeros((self._num_actions,), dtype=np.int8)
                    a[action] = 1
                    action = a

                env_state, reward, done, _ = self._env.step(action)

            i += 1

            # Render the environment if needed
            if self._render > 0 and self.total_episodes >= self._render:
                self._env.render()

            # Use the taskfile to modify reward and done
            additional_reward, additional_done = self._task(old_env_state, action, env_state)

            reward += additional_reward

            if additional_done is not None:
                done = additional_done

            # Add the reward of the action
            experience.reward = reward
            cumulative_reward += reward

            # Learn from the experience buffer
            if self._learn_freq == 0:
                do_learn = done
            else:
                do_learn = (self.total_timesteps % self._learn_freq == 0)

            if do_learn and not self._async_actor:
                s = datetime.datetime.now()
                d = (s - self._datetime).total_seconds()
                print('Start Learning, in-between is %.3f seconds...' % d)

                count = self._bdpi.train()
                ns = datetime.datetime.now()
                d = (ns - s).total_seconds()
                print('Learned %i steps in %.3f seconds, %.2f timesteps per second' % (count, d, count / d))
                sys.stderr.flush()
                sys.stdout.flush()
                self._datetime = ns

        return (env_state, cumulative_reward, done, i)

def async_loop(bdpi):
    """ Constantly ask BDPI to learn, used when --async-actor is set.
    """
    while True:
        bdpi.train()

def main():
    # Parse parameters
    parser = argparse.ArgumentParser(description="Reinforcement Learning for the Gym")

    parser.add_argument("--render", type=int, default=0, help="Enable a graphical rendering of the environment after N episodes")
    parser.add_argument("--monitor", action="store_true", default=False, help="Enable Gym monitoring for this run")
    parser.add_argument("--env", required=True, type=str, help="Gym environment to use")
    parser.add_argument("--retro", action='store_true', default=False, help="The environment is a OpenAI Retro environment (not a Gym one)")
    parser.add_argument("--atari", action="store_true", default=False, help="Wrap an Atari environment and use a more complex neural network")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes to run")
    parser.add_argument("--name", type=str, default='', help="Experiment name")
    parser.add_argument("--threads", type=int, default=1, help="Number of parallel processes used for training critics. Disables multiprocessing when set to 1")
    parser.add_argument("--async-actor", default=False, action="store_true", help="Learn in parallel with acting, useful for slow constant-rate environments")
    parser.add_argument("--taskfile", type=str, help="Name of a Python file that contains a task(s, a, s') -> reward, done, that determines the task to be learned by the agent")

    parser.add_argument("--erpoolsize", type=int, default=20000, help="Number of experiences stored by each option for experience replay")
    parser.add_argument("--er", type=int, default=256, help="Number of experiences used to build a replay minibatch")
    parser.add_argument("--erfreq", type=int, default=1, help="Learn using a batch of experiences every N time-steps, 0 for every episode")
    parser.add_argument("--loops", type=int, default=1, help="Number of replay batches replayed at each time-step")
    parser.add_argument("--aepochs", type=int, default=1, help="Number of epochs used to fit the actor")
    parser.add_argument("--cepochs", type=int, default=1, help="Number of epochs used to fit the critic")

    parser.add_argument("--cnn-type", default='atari', type=str, choices=['atari', 'mnist'], help="General shape of the CNN, if any. Either DQN-Like, or image-classification-like with more layers")
    parser.add_argument("--hidden", default=128, type=int, help="Hidden neurons of the policy network")
    parser.add_argument("--layers", default=1, type=int, help="Number of hidden layers in the networks")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate of the neural network")
    parser.add_argument("--load", type=str, help="File from which to load the neural network weights")
    parser.add_argument("--save", type=str, help="Basename of saved weight files. If not given, nothing is saved")

    parser.add_argument("--offpolicy-noise", action="store_true", default=False, help="Add some off-policy noise on the actions executed by the agent, using e-Greedy with --temp.")
    parser.add_argument("--pursuit-variant", type=str, choices=['generalized', 'ri', 'rp', 'pg'], default='rp', help="Pursuit Learning algorithm used")
    parser.add_argument("--learning-algo", type=str, choices=['egreedy', 'softmax', 'pursuit'], default='pursuit', help="Action selection method")
    parser.add_argument("--temp", type=str, default='0.1', help="Epsilon or temperature. Can be a value_factor format where value is multiplied by factor after every episode")
    parser.add_argument("--actor-count", type=int, default=1, help="Number of critics used by BDPI")
    parser.add_argument("--q-loops", type=int, default=1, help="Number of training iterations performed on the critic for each training epoch")
    parser.add_argument("--alr", type=float, default=0.05, help="Actor learning rate")
    parser.add_argument("--clr", type=float, default=0.2, help="Critic learning rate")

    args = parser.parse_args()

    # Loading task description file
    task = lambda s, a, snext: (0.0, None)

    if args.taskfile is not None:
        data = open(args.taskfile, 'r').read()
        compiled = compile(data, args.taskfile, 'exec')
        d = {}
        exec(compiled, d)

        if 'task' in d:
            task = d['task']

    # Instantiate learner
    learner = Learner(args, task)

    # Load weights if needed
    if args.load is not None:
        print('Loading', args.load)
        learner.loadstore(args.load, load=True)

    # Start async learner if needed
    if args.async_actor:
        t = threading.Thread(target=lambda: async_loop(learner._bdpi))
        t.start()

    # Execute the environment and learn from it
    f = open('out-' + args.name, 'w')
    print('# Arguments:', ' '.join(sys.argv[1:]), file=f)
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
            _, reward, done, length = learner.execute(learner._env.reset())

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
                    learner.loadstore(args.save, load=False)

                # Save last episode
                learner.save_episode(args.name + '-latest')

                old_dt = datetime.datetime.now()

            print(reward, avg, learner.total_timesteps, (datetime.datetime.now() - start_dt).total_seconds(), length, file=f)
            print(args.name, "Cumulative reward:", reward, "; average reward:", avg, "; length:", length)
            f.flush()
    finally:
        if args.monitor:
            learner._env.monitor.close()

        f.close()

        # Print timing statistics
        delta = datetime.datetime.now() - start_dt

        print('Learned during', str(delta).split('.')[0])
        print('Learning rate:', learner.total_timesteps / delta.total_seconds(), 'timesteps per second')

if __name__ == '__main__':
    main()
