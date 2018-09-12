import torch
import numpy as np

import time
import collections
import random
import copy
import sys
import os
import pickle
import lzo

ALPHA = 0.2
ALPHAA = 0.001
GAMMA = 0.99

class Experience:
    """ States, actions, rewards experienced by an agent
    """
    __slots__ = 'action', 'entropy', 'reward', 'next_experience', '_state'

    def __init__(self, state, action, entropy):
        self.action = action
        self.entropy = entropy

        self.reward = 0.0
        self.next_experience = None

        # Compress state when storing an experience
        s = state.tostring() # pickle.dumps(state)
        self._state = lzo.compress(s)

    def state(self):
        return Experience.unpack_state(self._state)

    @staticmethod
    def unpack_state(state_raw):
        return np.fromstring(lzo.decompress(state_raw), dtype=np.float32)


class Learner:
    """ Base learner class, used by the actor and critics. Async learners
        run in a separate process
    """

    def __init__(self, state_vars, num_actions, args, is_critic):
        self.state_vars = state_vars
        self.num_actions = num_actions
        self.args = args
        self.is_critic = is_critic

        self._setup()

    def state_dict(self):
        """ Return the state_dict of the model
        """
        return self._state_dict()

    def load_state_dict(self, s):
        """ Set the state of the model
        """
        self._load_state_dict(s)

    def _predict_model(self, model, inp):
        """ Return a Numpy prediction of a model
        """
        with torch.no_grad():
            return model[0](variable(inp)).data.cpu().numpy()

    def _train_model(self, model, inp, target, epochs):
        """ Train a model on inputs and outputs
        """
        v_inp = variable(inp)
        v_target = variable(target)

        # JIT the model if it is not yet jitted, and we know the final size of the input
        if isinstance(model[0], torch.nn.Sequential) and inp.shape[0] == self.args.er:
            print('JIT')
            model[0] = torch.jit.trace(v_inp)(model[0])

        # Perform training
        def closure():
            model[1].zero_grad()
            out = model[0](v_inp)
            loss = model[2](out, v_target)
            loss.backward()
            return loss

        for i in range(epochs):
            loss = model[1].step(closure)

        return loss

    def _state_dict(self):
        return self._models[0][0].state_dict()

    def _load_state_dict(self, s):
        for m in self._models:
            m[0].load_state_dict(s)

    def _setup(self):
        if self.is_critic:
            # Clipped DQN requires two models
            self._models = [self._make_model() for i in range(2)]
        else:
            # The actor only needs one model
            self._models = [self._make_model()]

    def _make_model(self):
        """ Create all the required network for a sub-option
        """
        def make_hidden(layers, inp_size):
            for i in range(self.args.layers):
                layers.append(torch.nn.Linear(inp_size if i == 0 else self.args.hidden, self.args.hidden))
                layers.append(torch.nn.Tanh())

        def make_model(layers):
            model = torch.nn.Sequential(*layers)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
            loss = torch.nn.MSELoss()

            return [model, optimizer, loss]

        layers = []

        make_hidden(layers, self.state_vars)
        layers.append(torch.nn.Linear(self.args.hidden, self.num_actions))

        if not self.is_critic:
            layers.append(torch.nn.Softmax(1))

        return make_model(layers)

class Actor(Learner):
    """ The actor learns using Conservative Policy Iteration from Q-Values provided
        to it. The actor is not a separate process
    """
    def __init__(self, state_vars, num_actions, args):
        super(Actor, self).__init__(state_vars, num_actions, args, False)

    def predict(self, state):
        """ Return a probability distribution over actions
        """
        return self._predict_model(self._models[0], state)[0]

    def train(self, states, actions, critic_qvalues):
        """ Train an actor using Policy Gradient or Pursuit
        """
        variant = self.args.pursuit_variant
        CN = np.arange(states.shape[0])

        # Pursuit: Value of the current state
        max_indexes = critic_qvalues.argmax(1)

        # Pursuit: Update actor
        train_probas = np.zeros_like(critic_qvalues)

        if variant == 'generalized':
            taken_action_qvalues = critic_qvalues[CN, actions][:, None]
            train_probas = (critic_qvalues > taken_action_qvalues).astype(np.float32)   # Pursue better actions
            train_probas[CN, max_indexes] = 1.0
        elif variant == 'ri':
            were_greedy = (max_indexes == actions).astype(np.float32)
            train_probas[CN, max_indexes] = were_greedy
        elif variant == 'rp':
            train_probas[CN, max_indexes] = 1.0

        # Normalize the direction to be pursued
        actor_probas = self._predict_model(self._models[0], states)

        # Discuss gradient vs target, and say that https://www.sciencedirect.com/science/article/pii/S0016003205000645 uses
        # a gradient-based approach with continuous actions (which sure works, it's policy gradient)
        train_probas /= 1e-6 + train_probas.sum(1)[:, None]
        train_probas = (1. - ALPHAA) * actor_probas + ALPHAA * train_probas

        # Fit the actor
        self._train_model(
            self._models[0],
            states,
            train_probas,
            self.args.epochs
        )

class Critic(Learner):
    """ A critic learned with Aggressive Bootstrapped Clipped DQN
    """
    def train(self, experiences):
        """ Train the critic from experiences. Call get_train_result() to obtain
            the (states, actions, qvalues) tuple to use to train the actor.
        """
        # Prepare Numpy arrays from the experiences (pickle is very slow at pickling experiences)
        states_raw = [e._state for e in experiences]
        actions = np.array([e.action for e in experiences], dtype=np.int32)
        rewards = np.array([e.reward for e in experiences], dtype=np.float32)

        # Prepare the list of states for which target Q-Values have to be computed
        next_experiences = [e.next_experience for e in experiences]
        nes = [e for e in next_experiences if e is not None]
        next_indexes = np.array([i for i, e in enumerate(next_experiences) if e is not None], dtype=np.int32)
        next_states_raw = [e._state for e in nes]

        data = (states_raw, actions, rewards, next_indexes, next_states_raw)
        self._data = data

    def get_train_result(self):
        return self._train(self._data)

    def predict(self, state):
        """ Return the Q-Values corresponding to a state
        """
        return self._predict(state)

    def _predict(self, state):
        return self._predict_model(self._models[0], state)[0]

    def _train(self, data):
        # Unpack data
        states_raw, actions, rewards, next_indexes, next_states_raw = data
        states_numpy = np.array([Experience.unpack_state(s) for s in states_raw])
        next_states = np.array([Experience.unpack_state(s) for s in next_states_raw])
        states = variable(states_numpy)

        for i in range(self.args.q_loops):
            # Q-Learning
            critic_qvalues = self._predict_model(self._models[0], states)       # Original Q-Values predicted by the newest Q-function

            self._train_loop(
                states,
                actions,
                rewards,
                critic_qvalues,
                next_states,
                next_indexes
            )

            # Clipped DQN, as Double DQN does, swaps the models after every training iteration
            self._models[0], self._models[1] = self._models[1], self._models[0]

        return (states_numpy, actions, critic_qvalues)

    def _train_loop(self, states, actions, rewards, critic_qvalues, next_states, next_indexes):
        """ Perform one iteration of Clipped DQN on the critic.
        """

        # Get all the next values, using the Clipped DQN target of min(Qa, Qb)
        QN = np.arange(states.shape[0])
        next_values = np.copy(rewards)

        next_values[next_indexes] += GAMMA * self._get_values(next_states)

        # Train using Q-Learning
        critic_qvalues[QN, actions] += ALPHA * (next_values - critic_qvalues[QN, actions])

        self._train_model(
            self._models[0],
            states,
            critic_qvalues,
            self.args.epochs
        )

    def _get_values(self, states):
        """ Return a list of values, one for each state.
        """
        qvalues_a = self._predict_model(self._models[0], states)
        qvalues_b = self._predict_model(self._models[1], states)
        QN = np.arange(states.shape[0])

        qvalues = np.minimum(qvalues_a, qvalues_b)  # Clipped DQN target

        return qvalues[QN, qvalues_a.argmax(1)]

class BDPI(object):
    """ The Bootstrapped Dual Policy Iteration algorithm.
    """
    def __init__(self, state_vars, num_actions, args, policy=None):
        """ Constructor.

            - state_vars: integer, number of state variables
            - num_actions: integer, number of actions available
            - args: Arguments from the command line, contains information about
                    learning rates, network shapes, etc.
            - policy: function(state) -> list of floats. If policy returns something
                    different than None, it overrides what the actor would have done
        """
        self.num_actions = num_actions
        self.policy = policy
        self.args = args
        self.use_actor = (args.learning_algo == 'pursuit')

        if '_' in args.temp:
            parts = args.temp.split('_')

            self._temp = float(parts[0])
            self._decay = float(parts[1])
        else:
            self._temp = float(args.temp)
            self._decay = 1.0

        self._experiences = collections.deque([], args.erpoolsize)
        self._actor_index = 0

        # Create actor and critic networks
        self._actor = Actor(state_vars, num_actions, self.args)
        self._critics = []

        for i in range(self.args.actor_count):
            self._critics.append(Critic(
                state_vars,
                num_actions,
                self.args,
                True                        # is_critic
            ))

    def loadstore(self, filename, load=True):
        """ Load the weights from a base filename
        """
        if load:
            self._actor.load_state_dict(torch.load(filename + '-actor'))

            for i, critic in enumerate(self._critics):
                weights = torch.load(filename + '-critic' + str(i))
                critic.load_state_dict(weights)
        else:
            torch.save(self._actor.state_dict(), filename + '-actor')

            for i, critic in enumerate(self._critics):
                torch.save(critic.state_dict(), filename + '-critic' + str(i))

    def reset(self, last_reward):
        self._temp *= self._decay

        # Choose the actor to be used now
        self._actor_index = random.randrange(self.args.actor_count)

    def _predict_probas(self, state):
        """ Return a list of unnormalized probabilities
        """
        state = state[None, :]

        if self.use_actor:
            # Get probas from the actor
            return self._actor.predict(state)
        else:
            # Get probas from the critic
            qvalues = self._critics[self._actor_index].predict(state)

            if self.args.learning_algo == 'egreedy':
                probas = np.zeros_like(qvalues)
                probas.fill(self._temp / (self.num_actions - 1))
                probas[qvalues.argmax()] = 1.0 - self._temp
            else:
                probas = np.exp((qvalues - qvalues.max()) / self._temp)

            return probas

    def train(self):
        # Sample experiences from the experience pool, restricting on the ones
        # that can be used with the current actor
        all_experiences = list(self._experiences)[:-1]
        count = min(len(all_experiences), self.args.er)

        if count < 16:
            return 0

        # Train each critic, then use its greedy function to train the actor
        critics = random.sample(self._critics, self.args.loops)

        for critic in critics:
            experiences = sample_wr(all_experiences, count)
            critic.train(experiences)
            states, actions, critic_qvalues = critic.get_train_result()

            if self.use_actor:
                self._actor.train(states, actions, critic_qvalues)

        return count

    def select_action(self, state, env_state):
        """ Return a sub-option to be executed and store an experience in the
            experience replay buffer.
        """
        probas = None

        if self.policy is not None:
            # Allow the policy to override what the option would have done
            probas = self.policy(env_state)

        if probas is None:
            probas = self._predict_probas(state)

        # Normalize the probabilities of best action to obtain the probabilities
        probas /= probas.sum()

        # Choose a model depending on a probability distribution
        action_index = np.random.multinomial(1, probas).argmax()
        entropy = -np.sum(probas * np.log2(probas))

        # Store the experience
        e = Experience(
            state,
            action_index,
            entropy
        )

        # Add the experience to a random set of actors.
        self._experiences.append(e)

        return action_index, e

###
# Utility functions
###
def sample_wr(population, k):
    "Chooses k random elements (with replacement) from a population"
    n = len(population)

    _random, _int = random.random, int  # speed hack 
    result = [None] * k

    for i in range(k):
        j = _int(_random() * n)
        result[i] = population[j]

    return result

def variable(inp):
    if torch.is_tensor(inp):
        rs = inp
    else:
        rs = torch.from_numpy(np.asarray(inp))

    return rs
