"""NFSP agents trained on the optimal stopping game."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
from absl import app
import pyspiel
import os

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.pytorch import nfsp
from open_spiel.python.games.optimal_stopping_game_config import OptimalStoppingGameConfig
from open_spiel.python.games.optimal_stopping_game_player_type import OptimalStoppingGamePlayerType
import random
import torch
import matplotlib.pyplot as plt
import pandas as pd
from open_spiel.python.games.optimal_stopping_game_util import OptimalStoppingGameUtil


def get_attacker_stage_policy_br(attacker_agent, num_states: int, num_actions: int, l: int, b: np.ndarray):
    """
    Extracts the attacker's stage policy from the NFSP agent

    :param attacker_agent: the NFSP attacker agent
    :param num_states: the number of states to consider for the stage policy
    :param num_actions: the number of actions to consider for the stage policy
    :param l: the number of stops left of the defender
    :param b: the current belief state
    """
    pi_2_stage = np.zeros((num_states+1, num_actions)).tolist()
    pi_2_stage[-1] = [0.5]*num_actions
    for s in range(num_states):
        o = {
            'info_state': [[l, b, b],
                           [l, b, s]],
            'legal_actions': [[],[0, 1]],
            'current_player': OptimalStoppingGamePlayerType.ATTACKER,
            "serialized_state": []
        }
        t_o= rl_environment.TimeStep(
            observations= o, rewards=None, discounts=None, step_type=None)
        pi_2_stage[s] = attacker_agent.step(t_o, is_evaluation=True).probs.tolist()
    return pi_2_stage


def get_stopping_probabilities(agents, l: int = 3):
    belief_space = np.linspace(0, 1, num=100)
    attacker_stopping_probabilities_no_intrusion = []
    attacker_stopping_probabilities_intrusion = []
    defender_stopping_probabilities = []
    for b in belief_space:
        info_state_intrusion = [[l, b, b], [l, b, 1]]
        info_state_no_intrusion = [[l, b, b], [l, b, 0]]
        defender_stopping_probabilities.append(agents[0]._act(info_state_intrusion[0], legal_actions = [0, 1])[1][1])
        attacker_stopping_probabilities_intrusion.append(agents[1]._act(info_state_intrusion[1], legal_actions = [0, 1])[1][1])
        attacker_stopping_probabilities_no_intrusion.append(agents[1]._act(info_state_no_intrusion[1], legal_actions = [0, 1])[1][1])

    attacker_stopping_probabilities_no_intrusion = round_vec(attacker_stopping_probabilities_no_intrusion)
    attacker_stopping_probabilities_intrusion = round_vec(attacker_stopping_probabilities_intrusion)
    defender_stopping_probabilities = round_vec(defender_stopping_probabilities)
    belief_space = round_vec(belief_space)

    return attacker_stopping_probabilities_intrusion, attacker_stopping_probabilities_no_intrusion, \
           defender_stopping_probabilities, belief_space


class NFSPPolicies(policy.Policy):
    """Joint policy to be evaluated."""

    def __init__(self, env, nfsp_policies, mode):
        game = env.game
        player_ids = [0, 1]
        super(NFSPPolicies, self).__init__(game, player_ids)
        self._policies = nfsp_policies
        self._mode = mode
        self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)

        self._obs["current_player"] = cur_player
        self._obs["info_state"][cur_player] = (
            state.information_state_tensor(cur_player))
        self._obs["legal_actions"][cur_player] = legal_actions

        info_state = rl_environment.TimeStep(
            observations=self._obs, rewards=None, discounts=None, step_type=None)

        with self._policies[cur_player].temp_mode_as(self._mode):
            p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict

def round_vecs(vecs):
    return list(map(lambda vec: list(map(lambda x: round(x, 2), vec)), vecs))

def round_vec(vec):
    return list(map(lambda x: round(x, 2), vec))

def main(unused_argv):
    params = OptimalStoppingGameConfig.default_params()
    params["use_beliefs"] = True
    params["T_max"] = 5


    params["R_SLA"] = 1
    params["R_ST"] = 2
    params["R_COST"] = -3
    params["R_INT"] = -3
    #params["L"] = 3
    params["obs_dist"] = " ".join(list(map(lambda x: str(x),[4/20,2/20,2/20,2/20,2/20,2/20,2/20,2/20,1/20,1/20,0])))
    params["obs_dist_intrusion"] = " ".join(list(map(lambda x: str(x),[1/20,1/20,2/20,2/20,2/20,2/20,2/20,2/20,2/20,4/20,0])))

    game = pyspiel.load_game("python_optimal_stopping_game_sequential", params)
    num_players = game.config.num_players

    env = rl_environment.Environment(game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    # network_parameters = {'batch_size': 256, 'hidden_layers_sizes': [64, 64, 64], 'memory_rl': 600000,
    #                       'memory_sl': 10000000.0, 'rl_learning_rate': 0.01, 'sl_learning_rate': 0.005}

    #network_parameters = {'batch_size': 512, 'hidden_layers_sizes': [512,512,512], 'memory_rl': 600000,
    #                       'memory_sl': 10000000.0, 'rl_learning_rate': 0.01, 'sl_learning_rate': 0.005}
    learn_every=128
    
    network_parameters = {'batch_size': 128, 'hidden_layers_sizes': [128], 'memory_rl': 200000, 'memory_sl': 2000000, 'rl_learning_rate': 0.1, 'sl_learning_rate': 0.005}

    # network_parameters = {'batch_size': 512, 'hidden_layers_sizes': [1024,1024,1024,1024,1024], 'memory_rl': 600000,
    #                       'memory_sl': 10000000.0, 'rl_learning_rate': 0.01, 'sl_learning_rate': 0.005}
    # learn_every=64

    # network_parameters = {'batch_size': 256, 'hidden_layers_sizes': [512,512,512,512], 'memory_rl': 600000,
    #                       'memory_sl': 10000000.0, 'rl_learning_rate': 0.01, 'sl_learning_rate': 0.005}
    # learn_every=64

    #network_parameters = {'batch_size': 256, 'hidden_layers_sizes': [512,512,512,512], 'memory_rl': 600000,
                         # 'memory_sl': 10000000.0, 'rl_learning_rate': 0.1, 'sl_learning_rate': 0.005}
    #learn_every=64

    #network_parameters = {'batch_size': 256, 'hidden_layers_sizes': [64, 64, 64], 'memory_rl': 600000, 'memory_sl': 10000000.0, 'rl_learning_rate': 0.01, 'sl_learning_rate': 0.005}
    hidden_layers_sizes = network_parameters['hidden_layers_sizes']
    batch_size = network_parameters['batch_size']
    rl_learning_rate = network_parameters['rl_learning_rate']
    sl_learning_rate = network_parameters['sl_learning_rate']
    memory_rl = network_parameters['memory_rl']
    memory_sl = network_parameters['memory_sl']
    

    # network_parameters = {'batch_size': 256, 'hidden_layers_sizes': [256,256,256], 'memory_rl': 600000,
    #                       'memory_sl': 10000000.0, 'rl_learning_rate': 0.1, 'sl_learning_rate': 0.005}
    #learn_every=64

    # network_parameters = {'batch_size': 512, 'hidden_layers_sizes': [1024,1024,1024,1024,1024], 'memory_rl': 600000,
    #                       'memory_sl': 10000000.0, 'rl_learning_rate': 0.007, 'sl_learning_rate': 0.001}
    # learn_every=64

    device_str="cuda:1"
    #device_str="cpu"
    #device_str="cuda:0"

    seed = 357
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # network_parameters = {'batch_size': 256, 'hidden_layers_sizes': [32, 32, 32], 'memory_rl': 60000,
    #                       'memory_sl': 1000000.0, 'rl_learning_rate': 0.01, 'sl_learning_rate': 0.009}

    #hidden_layers_sizes = network_parameters['hidden_layers_sizes']
    #batch_size = network_parameters['batch_size']
    #rl_learning_rate = network_parameters['rl_learning_rate']
    #sl_learning_rate = network_parameters['sl_learning_rate']
    #memory_rl = network_parameters['memory_rl']
    #memory_sl = network_parameters['memory_sl']

    expl_array  = []
    approx_expl_array = []
    ep_array = []
    game_value_array = []
    game_value_array_random = []
    game_value_array_heur = []

    eval_every = 10000
    #hidden_layers_sizes = [64, 64, 64]
    num_train_episodes = int(1e6)
    #num_train_episodes = int(350000)
    kwargs = {
        "replay_buffer_capacity": memory_rl,
        "epsilon_decay_duration": num_train_episodes,
        "epsilon_start": 0.06,
        "epsilon_end": 0.001,
        "lr_decay_duration": num_train_episodes,
        "lr_end": rl_learning_rate,
        "update_target_network_every": 300
    }

    agents = [
        nfsp.NFSP(player_id=idx,
                  state_representation_size = info_state_size,
                  num_actions = num_actions,
                  hidden_layers_sizes = hidden_layers_sizes,
                  reservoir_buffer_capacity = memory_sl,
                  anticipatory_param = 0.1,
                  batch_size = batch_size,
                  rl_learning_rate = rl_learning_rate,
                  sl_learning_rate = sl_learning_rate,
                  min_buffer_size_to_learn = 2000,
                  learn_every = learn_every,
                  stopping_game = True,
                  optimizer_str="adam",
                  device_str=device_str,
                  sl_lr_decay_duration=num_train_episodes,
                  sl_lr_end=sl_learning_rate,
                  **kwargs) for idx in range(num_players)
    ]

    expl_policies_avg = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

    
    for ep in range(num_train_episodes):
        if (ep + 1) % eval_every == 0 and ep+1 > 9000:

            # print("calculating approx expl..")
            # approxexpl = OptimalStoppingGameUtil.approx_exploitability(agents, env)
            # print("approx eplx = " + str(approxexpl[-1]))

            losses = [agent.loss for agent in agents]
            print("Calculating exact exploitability.. (Don't do this for large games!)")
            try:
                expl = exploitability.exploitability(env.game, expl_policies_avg)
            except Exception as e:
                expl = 0
                print(e)
                print("Some exception when calcluation exploitability")

            l=3
           
            print("Game value calculation:")
            
            game_value = OptimalStoppingGameUtil.game_value_MC(agents, env, defender_mode = nfsp.MODE.average_policy, \
                attacker_mode = nfsp.MODE.average_policy, use_defender_mode=True, use_attacker_mode= True)
            print("Current game value: " + str(game_value))
            game_value_against_random, game_value_against_heur = OptimalStoppingGameUtil.eval_defender_value(agents[0], env)
            
            game_value_array.append(game_value)
            game_value_array_random.append(game_value_against_random)
            game_value_array_heur.append(game_value_against_heur)

            print(f"Episode:{ep+1}, AVG Exploitability:{expl}, losses: {losses}")
            #print(f"l={l}, t={1}, Belief space: {belief_space}")
            #print(f"pi_2(S|b,0): {attacker_stopping_probabilities_no_intrusion}")
            #print(f"pi_2(S|b,1): {attacker_stopping_probabilities_intrusion}")
            #print(f"pi_1(S|b,-): {defender_stopping_probabilities}")
            sys.stdout.flush()

            expl_array.append(expl)
            # approx_expl_array.append(approxexpl[-1])
            ep_array.append(ep)


        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            time_step.observations["info_state"] = round_vecs(time_step.observations["info_state"])

            # print(f"time_step.rewards:{time_step.rewards}, player:{player_id}")
            action_output = agents[player_id].step(time_step)
            s = env.get_state

            # Update pi_2 if attacker
            if player_id == 1:
                s.update_pi_2(agents[player_id].pi_2_stage)

            action = [action_output.action]
            time_step = env.step(action)


        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)  
   

    evaluate_agents(agents, expl_array, game_value_array, game_value_array_random, game_value_array_heur)

    

def evaluate_agents(agents, expl_array, game_value_array, game_value_array_random, game_value_array_heur):

    experiment_no = 1

    attacker_stopping_probabilities_intrusion_3, attacker_stopping_probabilities_no_intrusion_3, \
           defender_stopping_probabilities_3, belief_space = get_stopping_probabilities(agents, 3)
    attacker_stopping_probabilities_intrusion_2, attacker_stopping_probabilities_no_intrusion_2, \
           defender_stopping_probabilities_2, belief_space = get_stopping_probabilities(agents, 2)
    attacker_stopping_probabilities_intrusion_1, attacker_stopping_probabilities_no_intrusion_1, \
           defender_stopping_probabilities_1, belief_space = get_stopping_probabilities(agents, 1)


    save_name = "Exploit_new_code_base_with_good_params_more_neurons" + str(experiment_no)

    if not os.path.isfile(save_name+".csv"):
        df = pd.DataFrame()
        df2 = pd.DataFrame()
        df["exploit " ] = expl_array
        df["value" ] = game_value_array
        df["game_value_array_random" ] = game_value_array_random
        df["game_value_array_heur" ] = game_value_array_heur
        
        df2["attacker_stopping_probabilities_intrusion_3"] = attacker_stopping_probabilities_intrusion_3
        df2["attacker_stopping_probabilities_no_intrusion_3"] = attacker_stopping_probabilities_no_intrusion_3
        df2["defender_stopping_probabilities_3"] = defender_stopping_probabilities_3

        df2["attacker_stopping_probabilities_intrusion_2"] = attacker_stopping_probabilities_intrusion_2
        df2["attacker_stopping_probabilities_no_intrusion_2"] = attacker_stopping_probabilities_no_intrusion_2
        df2["defender_stopping_probabilities_2"] = defender_stopping_probabilities_2

        df2["attacker_stopping_probabilities_intrusion_1"] = attacker_stopping_probabilities_intrusion_1
        df2["attacker_stopping_probabilities_no_intrusion_1"] = attacker_stopping_probabilities_no_intrusion_1
        df2["defender_stopping_probabilities_1"] = defender_stopping_probabilities_1

        df2["b_vec"] = belief_space

        #df["a_expl " +(str(network_parameters))] = approx_expl_array

        df.to_csv(save_name + ".csv", header='column_names')
        df2.to_csv(save_name + "_belief.csv", header='column_names')
        
    else: # else it exists so append without writing the header
        #print("hej")
        df = pd.read_csv(save_name+".csv")
        df2 = pd.DataFrame()
        df["exploit " ] = expl_array
        df["value"] = game_value_array
        df["game_value_array_random" ] = game_value_array_random
        df["game_value_array_heur" ] = game_value_array_heur
        
        df2["attacker_stopping_probabilities_intrusion_3"] = attacker_stopping_probabilities_intrusion_3
        df2["attacker_stopping_probabilities_no_intrusion_3"] = attacker_stopping_probabilities_no_intrusion_3
        df2["defender_stopping_probabilities_3"] = defender_stopping_probabilities_3

        df2["attacker_stopping_probabilities_intrusion_2"] = attacker_stopping_probabilities_intrusion_2
        df2["attacker_stopping_probabilities_no_intrusion_2"] = attacker_stopping_probabilities_no_intrusion_2
        df2["defender_stopping_probabilities_2"] = defender_stopping_probabilities_2

        df2["attacker_stopping_probabilities_intrusion_1"] = attacker_stopping_probabilities_intrusion_1
        df2["attacker_stopping_probabilities_no_intrusion_1"] = attacker_stopping_probabilities_no_intrusion_1
        df2["defender_stopping_probabilities_1"] = defender_stopping_probabilities_1

        df2["b_vec"] = belief_space

        #df["a_expl " +(str(network_parameters))] = approx_expl_array

        df.to_csv(save_name + ".csv", header='column_names')
        df2.to_csv(save_name + "_belief.csv", header='column_names')
   



if __name__ == "__main__":
    app.run(main)