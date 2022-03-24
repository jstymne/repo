import math
from typing import List, Tuple
import gym
import numpy as np
import threading
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from open_spiel.python.games.optimal_stopping_game_config_sequential import OptimalStoppingGameConfigSequential
from open_spiel.python.games.optimal_stopping_game_util import OptimalStoppingGameUtil


class OptimalStoppingGameApproxExp:
    """
    Class for computing exploitability of a strategy profile (pi_1, pi_2)
    """

    def __init__(self, pi_1, pi_2, config: OptimalStoppingGameConfigSequential, seed: int,
                 br_training_timesteps = 30000, br_evaluate_timesteps = 1000,
                 br_net_num_layers=3, br_net_num_hidden_neurons = 128, br_learning_rate = 3e-4, br_batch_size = 64,
                 br_steps_between_updates = 2048, br_training_device_str : str = "cpu"):
        """
        Initializes the object

        :param pi_1: the defender NFSP strategy
        :param pi_2: the attacker NFSP strategy
        :param config: the game configuration
        :param seed: the random seed
        :param br_training_timesteps: the number of time-steps to use when approximating best response strategies
        :param br_evaluate_timesteps: number of time-steps for evaluating a learned BR policy
        :param br_net_num_layers: number of hidden layers of the NN to learn an approximate BR
        :param br_net_num_hidden_neurons: number of hidden neurons per layer of the NN to learn an approximate BR
        :param br_learning_rate: the learning rate for learning approximate best response strategies
        :param br_batch_size: the batch size for learning best response strategies
        :param br_steps_between_updates: the number of steps between each update during learning of best response strategies
        :param br_training_device_str: the device for the training of BR strategies
        """
        self.pi_1 = pi_1
        self.pi_2 = pi_2
        self.config = config
        self.attacker_mdp = self._get_attacker_mdp()
        self.defender_pomdp = self._get_defender_pomdp()
        self.seed = seed
        self.br_timesteps = br_training_timesteps
        self.br_evaluate_timesteps = br_evaluate_timesteps
        self.br_net_num_layers = br_net_num_layers
        self.br_net_num_hidden_neurons = br_net_num_hidden_neurons
        self.br_learning_rate = br_learning_rate
        self.br_batch_size = br_batch_size
        self.br_steps_between_updates = br_steps_between_updates
        self.br_training_device_str = br_training_device_str

    def _get_attacker_mdp(self) -> gym.Env:
        """
        :return: the attacker MDP for calculating a best response strategy
        """
        env = StoppingGameAttackerMDPEnv(config=self.config, pi_1=self.pi_1, pi_2=self.pi_2)
        return env

    def _get_defender_pomdp(self) -> gym.Env:
        """
        :return: the defender POMDP for calculating a best response strategy
        """
        env = StoppingGameDefenderPOMDPEnv(config=self.config, pi_1=self.pi_1, pi_2=self.pi_2)
        return env

    def approx_exploitability(self) -> float:
        """
        :return: approximate exploitability of pi_1 and pi_2
        """
        print("--- Calculating approximate exploitability ---")
        avg_attacker_br_R = self.attacker_br_avg_reward()
        avg_defender_br_R = self.defender_br_avg_reward()
        approx_expl = abs(avg_attacker_br_R + avg_defender_br_R)/2
        return approx_expl

    def attacker_br_avg_reward(self) -> float:
        """
        Learns an approximate best response strategy of the attacker and returns its average reward

        :return: the average reward of the approximate best response strategy
        """
        policy_kwargs = dict(net_arch=[self.br_net_num_hidden_neurons]*self.br_net_num_layers)
        # log_dir = "./"
        env = Monitor(self.attacker_mdp)
        # env = self.attacker_mdp
        model = PPO("MlpPolicy", env, verbose=0,
                    policy_kwargs=policy_kwargs, n_steps=self.br_steps_between_updates,
                    batch_size=self.br_batch_size, learning_rate=self.br_learning_rate, seed=self.seed,
                    device=self.br_training_device_str, gamma=1)
        self.attacker_mdp.ppo_pi_2 = model
        print(" ** Starting training of an approximate best response strategy of the attacker ** ")
        progress_thread = ProgressThread(env=env, max_steps=self.br_timesteps)
        progress_thread.start()
        model.learn(total_timesteps=self.br_timesteps)
        progress_thread.running = False
        print("** Training of an approximate best response strategy of the attacker complete **")

        obs = env.reset()
        r = 0
        returns = []
        for i in range(self.br_evaluate_timesteps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            r += reward
            if done:
                returns.append(r)
                r = 0
                obs = env.reset()
        avg_R = -np.mean(returns)
        print("Attacker approximate best response AVG Return:{}".format(avg_R))
        return float(avg_R)


    def defender_br_avg_reward(self) -> float:
        """
        Learns an approximate best response strategy of the defender and returns its average reward

        :return: the average reward of the approximate best response strategy
        """
        policy_kwargs = dict(net_arch=[self.br_net_num_hidden_neurons]*self.br_net_num_layers)
        # log_dir = "./"
        env = Monitor(self.defender_pomdp)
        # env = self.defender_pomdp
        model = PPO("MlpPolicy", env, verbose=0,
                    policy_kwargs=policy_kwargs, n_steps=self.br_steps_between_updates,
                    batch_size=self.br_batch_size, learning_rate=self.br_learning_rate, seed=self.seed,
                    device=self.br_training_device_str, gamma=1)
        print("** Starting training of an approximate best response strategy of the defender **")
        progress_thread = ProgressThread(env=env, max_steps=self.br_timesteps)
        progress_thread.start()
        model.learn(total_timesteps=self.br_timesteps)
        progress_thread.running = False
        print("** Training of an approximate best response strategy of the defender complete **")

        obs = env.reset()
        r = 0
        returns = []
        for i in range(self.br_evaluate_timesteps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            r += reward
            if done:
                returns.append(r)
                r = 0
                obs = env.reset()
        avg_R = np.mean(returns)
        print("Defender approximate best response AVG Return:{}".format(avg_R))
        return float(avg_R)


class StoppingGameAttackerMDPEnv(gym.Env):
    """
    MDP where the attacker faces a static defender policy. The optimal policy in this MDP is a best response strategy
    of the attacker
    """

    def __init__(self, config: OptimalStoppingGameConfigSequential, pi_1, pi_2):
        """
        Initializes the environment

        :param config: the environment configuration
        :param pi_1: NFSP policy of the defender
        :param pi_2: NFSP policy of the attacker
        """
        self.config = config
        self.l = config.L
        self.s0 = 0
        self.b0 = config.initial_belief
        self.pi_1 = pi_1
        self.pi_2 = pi_2
        self.observation_space = gym.spaces.Box(low=np.array([0,0,0]), high=np.array([self.config.L,1,2]), dtype=np.float32,
                                                shape=(3,))
        self.action_space = gym.spaces.Discrete(2)
        self.num_actions = 2
        self.t = 0
        self.ppo_pi_2 = None

    def get_attacker_dist(self, obs):
        obs = np.array([obs])
        actions, values, log_prob = self.ppo_pi_2.policy.forward(obs=torch.tensor(obs))
        action = actions[0]
        if action == 1:
            stop_prob = math.exp(log_prob)
        else:
            stop_prob = 1-math.exp(log_prob)
        return [1-stop_prob, stop_prob]

    def get_attacker_stage_policy_avg(self) -> List:
        """
        Extracts the stage policy from pi_2

        :return: the attacker's stage policy
        """
        # actions, values, log_prob = self.ppo_pi_2.policy.forward(obs=torch.tensor(np.array([[self.l,self.b[1],1]])))
        # action = actions[0]
        # if action == 1:
        #     stop_prob = math.exp(log_prob)
        # else:
        #     stop_prob = 1-math.exp(log_prob)

        pi_2_stage = np.zeros((3, 2)).tolist()
        pi_2_stage[-1] = [0.5]*2
        for s in range(2):
            o = [self.l,self.b[1],s]
            # pi_2_stage[s] = self.pi_2._act(o, legal_actions = [0,1])[1]
            pi_2_stage[s] = self.get_attacker_dist(obs=o)
        return pi_2_stage

    def step(self, a2) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Takes a step in the environment

        :param a2: the attacker's action
        :return: o, r, done, info
        """
        done = False
        a1 = self.defender_action()
        r = -self.config.R[self.l-1][a1][a2][self.s]
        T = self.config.T[self.l-1]
        old_s = self.s
        self.s = self.sample_next_state(a1=a1, a2=a2, T=T)
        o = max(self.config.O)
        old_b = self.b
        pi_2_stage = None
        if self.s == 2 or self.t >= self.config.T_max:
            done = True
        else:
            o = self.sample_next_observation(a1=a1, a2=a2)
            pi_2_stage = self.get_attacker_stage_policy_avg()
            self.b = OptimalStoppingGameUtil.next_belief(o=o, a1=a1, b=self.b, pi_2=pi_2_stage,
                                                         config=self.config, l=self.l, a2=a2)
        old_l = self.l
        self.l = self.l-a1
        info = {"o": o, "s": self.s}
        self.t += 1
        # print(f"a1:{a1}, a2:{a2}, old_s:{old_s}, new_s:{self.s}, old_l: {old_l}, l.{self.l}, old_b:{old_b[1]}, new_b:{self.b[1]}, r:{r}, pi_2_stage:{pi_2_stage}, o:{o}")
        return np.array([self.l, self.b[1], self.s]), r, done, info

    def defender_action(self) -> int:
        """
        Samples a defender action from a static policy

        :return: the sampled defender action
        """
        stop_prob = self.pi_1._act([self.l, self.b[1], self.b[1]], legal_actions = [0, 1])[1][1]
        # if self.b[1] >= 0.8:
        #     print(f"defender stop_prob:{stop_prob}, b1:{self.b[1]}, input:{[self.l, self.b[1], self.b[1]]}")
        if np.random.rand() <= stop_prob:
            return 1
        else:
            return 0

    def sample_next_state(self, a1: int, a2: int, T: np.ndarray) -> int:
        """
        Samples the next state

        :param a1: action of the defender
        :param a2: action of the attacker
        :param T: the transition tensor
        :return: the next state
        """
        state_probs = []
        for s_prime in self.config.S:
            state_probs.append(T[a1][a2][self.s][s_prime])
        s_prime = np.random.choice(np.arange(0, len(self.config.S)), p=state_probs)
        return s_prime

    def sample_next_observation(self, a1: int, a2: int) -> int:
        """
        Samples the next observation

        :param a1: the action of the defender
        :param a2: the action of the attacker
        :return: the next observation
        """
        observation_probs = []
        for o in self.config.O:
            observation_probs.append(self.config.Z[a1][a2][self.s][o])
        o = np.random.choice(np.arange(0, len(self.config.O)), p=observation_probs)
        return o

    def reset(self) -> np.ndarray:
        """
        Resets the environment

        :return: the initial observation
        """
        self.s = 0
        self.b = self.config.initial_belief
        self.l = self.config.L
        self.t = 0
        return np.array([self.l, self.b0[1], self.s])

    def render(self):
        raise NotImplementedError("not supported")



class StoppingGameDefenderPOMDPEnv(gym.Env):
    """
    POMDP where the defender faces a static attacker policy. The optimal policy in this POMDP is a
    best response strategy of the defender
    """

    def __init__(self, config: OptimalStoppingGameConfigSequential, pi_1, pi_2):
        """
        Initializes the game

        :param config: the game configuration
        :param pi_1: the defender NFSP policy
        :param pi_2: the attacker NFSP policy
        """
        self.config = config
        self.l = config.L
        self.s0 = 0
        self.b0 = config.initial_belief
        self.pi_1 = pi_1
        self.pi_2 = pi_2
        self.t = 0
        self.observation_space = gym.spaces.Box(low=np.array([0,0,0]), high=np.array([self.config.L,1,1]), dtype=np.float32,
                                                shape=(3,))
        self.action_space = gym.spaces.Discrete(2)


    def get_attacker_stage_policy_avg(self):
        """
        Extracts the stage policy from pi_2

        :return: the attacker's stage policy
        """
        pi_2_stage = np.zeros((3, 2)).tolist()
        pi_2_stage[-1] = [0.5]*2
        for s in range(2):
            o = [self.l,self.b[1],s]
            pi_2_stage[s] = self.pi_2._act(o, legal_actions = [0,1])[1]
        return pi_2_stage

    def step(self, a1) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Steps the environment

        :param a1: the defender action
        :return: o, r, done, info
        """
        done = False
        a2 = self.attacker_action()
        r = self.config.R[self.l-1][a1][a2][self.s]
        T = self.config.T[self.l-1]
        self.s = self.sample_next_state(a1=a1, a2=a2, T=T)
        o = max(self.config.O)
        if self.s == 2 or self.t >= self.config.T_max:
            done = True
        else:
            o = self.sample_next_observation(a1=a1, a2=a2)
            pi_2_stage = self.get_attacker_stage_policy_avg()
            self.b = OptimalStoppingGameUtil.next_belief(o=o, a1=a1, b=self.b, pi_2=pi_2_stage,
                                                         config=self.config, l=self.l, a2=a2)
        self.l = self.l-a1
        info = {"o": o, "s": self.s}
        self.t += 1
        return np.array([self.l, self.b[1], self.b[1]]), r, done, info

    def attacker_action(self) -> int:
        """
        Samples an attacker action from a static policy

        :return: the sampled attacker action
        """
        stop_prob = self.pi_2._act([self.l, self.b[1], self.s], legal_actions = [0, 1])[1][1]
        if np.random.rand() < stop_prob:
            return 1
        else:
            return 0

    def sample_next_state(self, a1: int, a2: int, T: np.ndarray) -> int:
        """
        Samples the next state

        :param a1: action of the defender
        :param a2: action of the attacker
        :param T: the transition tensor
        :return: the next state
        """
        state_probs = []
        for s_prime in self.config.S:
            state_probs.append(T[a1][a2][self.s][s_prime])
        s_prime = np.random.choice(np.arange(0, len(self.config.S)), p=state_probs)
        return s_prime

    def sample_next_observation(self, a1: int, a2: int) -> int:
        """
        Samples the next observation

        :param a1: the action of the defender
        :param a2: the action of the attacker
        :return: the next observation
        """
        observation_probs = []
        for o in self.config.O:
            observation_probs.append(self.config.Z[a1][a2][self.s][o])
        o = np.random.choice(np.arange(0, len(self.config.O)), p=observation_probs)
        return o

    def reset(self) -> np.ndarray:
        """
        Resets the environment

        :return: the initial observation
        """
        self.s = 0
        self.b = self.config.initial_belief
        self.l = self.config.L
        self.t = 0
        return np.array([self.l, self.b0[1], self.s])

    def render(self):
        raise NotImplementedError("not supported")


class ProgressThread(threading.Thread):

    def __init__(self, env, max_steps):
        threading.Thread.__init__(self)
        self.env = env
        self.max_steps = max_steps
        self.running = True

    def run(self) -> None:
        while self.running:
            time.sleep(5)
            if self.running:
                print(f"Learning a best response strategy, progress:{int(100*round(self.env.total_steps/self.max_steps,2))}%")
