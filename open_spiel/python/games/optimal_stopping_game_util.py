from typing import List
import numpy as np
from pyrsistent import v
from open_spiel.python import rl_environment
from open_spiel.python.games.optimal_stopping_game_config import OptimalStoppingGameConfig
from open_spiel.python.games.optimal_stopping_game_observation_type import OptimalStoppingGameObservationType
from open_spiel.python.pytorch import nfsp
import matplotlib.pyplot as plt

class OptimalStoppingGameUtil:

    @staticmethod
    def next_state(state: int, defender_action: int, attacker_action: int, l: int) -> int:
        """
        Computes the next state of the game given the current state and a defender action and an attacker action
        :param state: the current state
        :param defender_action: the action of the defender
        :param attacker_action: the action of the attacker
        :param l: the number of stops remaining
        :return: the next state
        """

        # Terminal state already
        if state == 2:
            return 2

        # Attacker aborts
        if state ==1  and attacker_action == 1:
            return 2

        # Defender final stop
        if defender_action == 1 and l == 1:
            return 2

        # Intrusion starts
        if state == 0 and attacker_action == 1:
            return 1

        # Stay in the current state
        return state

    @staticmethod
    def reward_function(state: int, defender_action: int, attacker_action: int, l: int,
                        config: OptimalStoppingGameConfig):
        """
        Computes the defender reward (negative of attacker reward)

        :param state: the state of the game
        :param defender_action: the defender action
        :param attacker_action: the attacker action
        :param l: the number of stops remaining
        :param config: the game config
        :return: the reward
        """
        # Terminal state
        if state == 2:
            return 0

        # No intrusion state
        if state == 0:
            # Continue and Wait
            if defender_action == 0 and attacker_action == 0:
                return config.R_SLA
            # Continue and Attack
            if defender_action == 0 and attacker_action == 1:
                return config.R_SLA
            # Stop and Wait
            if defender_action == 1 and attacker_action == 0:
                return config.R_COST / config.L
            # Stop and Attack
            if defender_action == 1 and attacker_action == 1:
                return config.R_COST / config.L + config.R_ST / l

        # Intrusion state
        if state == 1:
            # Continue and Continue
            if defender_action == 0 and attacker_action == 0:
                return config.R_SLA + config.R_INT
            # Continue and Stop
            if defender_action == 0 and attacker_action == 1:
                return config.R_SLA
            # Stop and Continue
            if defender_action == 1 and attacker_action == 0:
                return config.R_COST / config.L + config.R_ST / l
            # Stop and Stop
            if defender_action == 1 and attacker_action == 1:
                return config.R_COST / config.L

        raise ValueError("Invalid input, s:{}, a1:{}, a2:{}".format(state, defender_action, attacker_action))


    @staticmethod
    def get_observation_type(obs: int, config: OptimalStoppingGameConfig) -> OptimalStoppingGameObservationType:
        """
        Returns the type of the observation

        :param obs: the observation to get the type of
        :return: observation type
        """
        if obs == max(config.obs):
            return OptimalStoppingGameObservationType.TERMINAL
        if obs == max(config.obs):
            return OptimalStoppingGameObservationType.TERMINAL
        else:
            return OptimalStoppingGameObservationType.NON_TERMINAL


    @staticmethod
    def bayes_filter(s_prime: int, o: int, a1: int, b: List, pi_2: List, config: OptimalStoppingGameConfig, l: int) -> float:
        """
        A Bayesian filter to compute the belief of player 1
        of being in s_prime when observing o after taking action a in belief b given that the opponent follows
        strategy pi_2

        :param s_prime: the state to compute the belief of
        :param o: the observation
        :param a1: the action of player 1
        :param b: the current belief point
        :param pi_2: the policy of player 2
        :param l: stops remaining
        :param config: the game config
        :return: b_prime(s_prime)
        """
        l=l-1
        norm = 0
        #print("belief " + str(b))
        #print("pi_2 " + str(pi_2))
        #print(l)
        #print(config.T[2][:][1][1][:])
        for s in config.S:
            for a2 in config.A2:
                #print("a2" + str(a2))
                for s_prime_1 in config.S:
                    
                    prob_1 = config.Z[a1][a2][s_prime_1][o]
                    rest = b[s]*config.T[l][a1][a2][s][s_prime_1]*pi_2[s][a2]
                    #print("prob1 " + str(prob_1))
                    #print("rest " + str(rest))
                    #if (prob_1 == 0 and rest == 1):
                        #print("l " + str(l))
                        #print("s " +str(s))
                        #print("sprime" + str(s_prime_1))
                        #print("a1 " + str(a1))
                        #print("a2 " + str(a2))
                        #print("obs " + str(o))
                        #print(config.Z)
                    
                    norm += b[s]*prob_1*config.T[l][a1][a2][s][s_prime_1]*pi_2[s][a2]
            print(f"error, b_prime:{b_prime}, o:{o}, a1:{a1}, b:{b}, pi_2:{pi_2}, "
                  f"a2: {a2}, s:{s}")
        assert round(sum(b_prime), 2) == 1
        return b_prime

    # @staticmethod
    # def approx_exploitability(agents, env):
    #
    #     mc_episodes = 1000
    #     #Calculation v_1 which is the expected value of defender BR vs attacker average
    #     v_1 = [0,0]
    #     v1_vec = []
    #     for ep in range(mc_episodes):
    #         time_step = env.reset()
    #         while not time_step.last():
    #             player_id = time_step.observations["current_player"]
    #             time_step.observations["info_state"] = OptimalStoppingGameUtil.round_vec(time_step.observations["info_state"])
    #
    #             #best_response for defender
    #             if player_id == 0:
    #                 with agents[player_id].temp_mode_as(nfsp.MODE.best_response):
    #                     action_output = agents[player_id].step(time_step, is_evaluation=True)
    #             #average policy for attacker
    #             if player_id == 1:
    #                 with agents[player_id].temp_mode_as(nfsp.MODE.average_policy):
    #                     action_output = agents[player_id].step(time_step, is_evaluation=True)
    #
    #             s = env.get_state
    #             action = [action_output.action]
    #             time_step = env.step(action)
    #             #print(time_step)
    #             #Update pi2
    #             time_step.observations["info_state"] = OptimalStoppingGameUtil.round_vec(time_step.observations["info_state"])
    #             current_l = time_step.observations["info_state"][0][0]
    #             current_belief = time_step.observations["info_state"][0][1]
    #             new_pi_2 = OptimalStoppingGameUtil.update_pi_2(agents[1],current_belief,current_l, temp_mode=nfsp.MODE.average_policy, is_temp_mode=True)
    #             s.update_pi_2(new_pi_2)
    #
    #
    #         #Episode over
    #         agents[0].prep_next_episode_MC(time_step)
    #
    #         agents[1].prep_next_episode_MC(time_step)
    #
    #         v_1 = v_1 + s.returns()
    #         v1_vec.append(v_1[0] / (ep+1))
    #
    #     v_1 = v_1 / mc_episodes
    #     #Calculation v_2 which is the expected value of defender average vs attacker BR
    #     v_2 = [0,0]
    #     v2_vec = []
    #     for ep in range(mc_episodes):
    #         time_step = env.reset()
    #         while not time_step.last():
    #             player_id = time_step.observations["current_player"]
    #             time_step.observations["info_state"] = OptimalStoppingGameUtil.round_vec(time_step.observations["info_state"])
    #
    #             #average policy for defender
    #             if player_id == 0:
    #                 with agents[player_id].temp_mode_as(nfsp.MODE.average_policy):
    #                     action_output = agents[player_id].step(time_step, is_evaluation=True)
    #             #BR policy for attacker
    #             if player_id == 1:
    #                 with agents[player_id].temp_mode_as(nfsp.MODE.best_response):
    #                     action_output = agents[player_id].step(time_step, is_evaluation=True)
    #
    #             s = env.get_state
    #             action = [action_output.action]
    #             time_step = env.step(action)
    #             time_step.observations["info_state"] = OptimalStoppingGameUtil.round_vec(time_step.observations["info_state"])
    #             current_l = time_step.observations["info_state"][0][0]
    #             current_belief = time_step.observations["info_state"][0][1]
    #             new_pi_2 = OptimalStoppingGameUtil.update_pi_2(agents[1],current_belief,current_l, temp_mode=nfsp.MODE.best_response, is_temp_mode=True)
    #             s.update_pi_2(new_pi_2)
    #
    #
    #         #Episode over
    #         agents[0].prep_next_episode_MC(time_step)
    #         agents[1].prep_next_episode_MC(time_step)
    #
    #         v_2 = v_2 + s.returns()
    #         v2_vec.append(v_2[0] / (ep+1))
    #     v_2 = v_2 / mc_episodes
    #     return np.subtract(v1_vec,v2_vec)
    #
    # @staticmethod
    # def round_vec(vecs):
    #     return list(map(lambda vec: list(map(lambda x: round(x, 2), vec)), vecs))
