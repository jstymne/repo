from typing import List
import numpy as np
from open_spiel.python.games.optimal_stopping_game_config import OptimalStoppingGameConfig
from open_spiel.python.games.optimal_stopping_game_observation_type import OptimalStoppingGameObservationType
from open_spiel.python.games.optimal_stopping_game_eval_agent import OptimalStoppingEvalAgent
import matplotlib.pyplot as plt
from open_spiel.python.pytorch import nfsp

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
        for s in config.S:
            for a2 in config.A2:
                for s_prime_1 in config.S:
                    prob_1 = config.Z[a1][a2][s_prime_1][o]
                    norm += b[s]*prob_1*config.T[l][a1][a2][s][s_prime_1]*pi_2[s][a2]

        if norm == 0:
            return 0
        temp = 0

        for s in config.S:
            for a2 in config.A2:
                temp += config.Z[a1][a2][s_prime][o]*config.T[l][a1][a2][s][s_prime]*b[s]*pi_2[s][a2]

        b_prime_s_prime = temp/norm
        assert b_prime_s_prime <=1
        if s_prime == 2 and o != config.O[-1]:
            assert b_prime_s_prime <= 0.01
        return b_prime_s_prime

    @staticmethod
    def p_o_given_b_a1_a2(o: int, b: List, a1: int, a2: int, config: OptimalStoppingGameConfig) -> float:
        """
        Computes P[o|a,b]
        :param o: the observation
        :param b: the belief point
        :param a1: the action of player 1
        :param a2: the action of player 2
        :param config: the game config
        :return: the probability of observing o when taking action a in belief point b
        """
        prob = 0
        for s in config.S:
            for s_prime in config.S:
                prob += b[s] * config.T[a1][a2][s][s_prime] * config.Z[a1][a2][s_prime][o]
        assert prob < 1
        return prob

    @staticmethod
    def next_belief(o: int, a1: int, b: List, pi_2: List, config: OptimalStoppingGameConfig, l: int,
                    a2 : int = 0, s : int = 0) -> List:
        """
        Computes the next belief using a Bayesian filter
        :param o: the latest observation
        :param a1: the latest action of player 1
        :param b: the current belief
        :param pi_2: the policy of player 2
        :param config: the game config
        :param l: stops remaining
        :param a2: the attacker action (for debugging, should be consistent with pi_2)
        :param s: the true state (for debugging)
        :return: the new belief
        """
        b_prime = np.zeros(len(config.S))
        for s_prime in config.S:
            b_prime[s_prime] = OptimalStoppingGameUtil.bayes_filter(s_prime=s_prime, o=o, a1=a1, b=b,
                                                                    pi_2=pi_2, config=config, l=l)
        if round(sum(b_prime), 2) != 1:
            print(f"error, b_prime:{b_prime}, o:{o}, a1:{a1}, b:{b}, pi_2:{pi_2}, "
                  f"a2: {a2}, s:{s}")
        assert round(sum(b_prime), 2) == 1
        return b_prime

       
    @staticmethod
    def round_vec(vecs):
         return list(map(lambda vec: list(map(lambda x: round(x, 2), vec)), vecs))
    
    @staticmethod
    def round_vecs(vecs):
        return list(map(lambda vec: list(map(lambda x: round(x, 2), vec)), vecs))

    @staticmethod
    def approx_exploitability(agents, env):
        v_1 = OptimalStoppingGameUtil.game_value_MC(agents, env, defender_mode = nfsp.MODE.best_response, \
            attacker_mode = nfsp.MODE.average_policy, use_defender_mode=True, use_attacker_mode= True)
        
        v_2 = OptimalStoppingGameUtil.game_value_MC(agents, env, defender_mode = nfsp.MODE.average_policy, \
            attacker_mode = nfsp.MODE.best_response, use_defender_mode=True, use_attacker_mode= True)
        
        return (v_1-v_2)/2

    @staticmethod
    def game_value_MC(agents, env, defender_mode = None, attacker_mode = None, use_defender_mode = False, use_attacker_mode = False):
        mc_episodes = 1000

        #print(time_step)
        #Calculation of v

        v = [0,0]
        v_vec = []
        for ep in range(mc_episodes):
            
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                time_step.observations["info_state"] = OptimalStoppingGameUtil.round_vecs(time_step.observations["info_state"])


                if use_defender_mode and player_id == 0:
                    with agents[player_id].temp_mode_as(defender_mode):
                        action_output = agents[player_id].step(time_step, is_evaluation=True)
                
                elif use_attacker_mode and player_id == 1:
                    with agents[player_id].temp_mode_as(attacker_mode):
                        action_output = agents[player_id].step(time_step, is_evaluation=True)
                
                else: #Else dont use mode for that player
                # print(f"time_step.rewards:{time_step.rewards}, player:{player_id}")
                    action_output = agents[player_id].step(time_step, is_evaluation = True)
                
                
                
                s = env.get_state

                # Update pi_2 if attacker
                if player_id == 1:
                    s.update_pi_2(agents[player_id].pi_2_stage)

                action = [action_output.action]
                time_step = env.step(action)

            # Episode is over, step all agents with final info state.
            agents[0].prep_next_episode_MC(time_step)
            agents[1].prep_next_episode_MC(time_step)

            #Episode over
            v = v + s.returns()
            v_vec.append(v[0] / (ep+1))
        #print(v_1)

        v = v / mc_episodes

        #print("Current game value is " + str(v[0]))
        

        #print(v_vec)
        #plt.plot(v_vec)
        #plt.ylabel('some numbers')
        #plt.show()
        #plt.savefig('foo.png')


        return v[0]

    @staticmethod
    def eval_defender_value(defender_agent, env):
        print("Calculating game value against random attacker")
        random_agent = OptimalStoppingEvalAgent(player_id = 1, num_actions = 2, evaluation_type = "RandomAttacker")
        value_against_random = OptimalStoppingGameUtil.game_value_MC(agents = [defender_agent, random_agent], env = env, \
            defender_mode = nfsp.MODE.average_policy, use_attacker_mode= False, use_defender_mode=True)

        print("Value against random attacker: " + str(value_against_random))

        print("Calculating game value against heuristic attacker")
        heur_agent = OptimalStoppingEvalAgent(player_id = 1, num_actions = 2, evaluation_type = "HeuristicAttacker")
        value_against_heur = OptimalStoppingGameUtil.game_value_MC(agents = [defender_agent, heur_agent], env = env, \
            defender_mode = nfsp.MODE.average_policy, use_attacker_mode= False, use_defender_mode=True)
        print("Value against heuristic attacker: " + str(value_against_heur))

        return value_against_random, value_against_heur
