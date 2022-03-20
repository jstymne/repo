# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NFSP agents trained on the optimal stopping game."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from absl import app
from absl import logging
import pyspiel
import os

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.pytorch import nfsp
from open_spiel.python.games.optimal_stopping_game_config import OptimalStoppingGameConfig
from open_spiel.python.games.optimal_stopping_game_util import OptimalStoppingGameUtil

import matplotlib.pyplot as plt
import pandas as pd 
import random


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

def round_vec(vecs):
    return list(map(lambda vec: list(map(lambda x: round(x, 2), vec)), vecs))

def main(unused_argv):
    params = OptimalStoppingGameConfig.default_params()
    params["use_beliefs"] = True
    params["T_max"] = 5

    
    params["R_SLA"] = 0
    params["R_ST"] = 2
    params["R_COST"] = -1
    params["R_INT"] = -2

    #Test different distributions
    #params["obs_dist"] = " ".join(list(map(lambda x: str(x),[3/20,3/20,3/20,3/20,2/20,2/20,1/20,1/20,1/20,1/20,0])))
    #params["obs_dist_intrusion"] = " ".join(list(map(lambda x: str(x),[1/20,1/20,1/20,1/20,2/20,2/20,3/20,3/20,3/20,3/20,0])))


    #Uniform dist
    #params["obs_dist"] = " ".join(list(map(lambda x: str(x),[2/20,2/20,2/20,2/20,2/20,2/20,2/20,2/20,2/20,2/20,0])))
    #params["obs_dist_intrusion"]  = " ".join(list(map(lambda x: str(x),[2/20,2/20,2/20,2/20,2/20,2/20,2/20,2/20,2/20,2/20,0])))

    game = pyspiel.load_game("python_optimal_stopping_game", params)
    num_players = game.config.num_players
    
    #game = pyspiel.convert_to_turn_based(game)
    env = rl_environment.Environment(game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    
    network_parameters = {'batch_size': 256, 'hidden_layers_sizes': [64, 64, 64], 'memory_rl': 600000, 'memory_sl': 10000000.0, 'rl_learning_rate': 0.01, 'sl_learning_rate': 0.005}
    hidden_layers_sizes = network_parameters['hidden_layers_sizes']
    batch_size = network_parameters['batch_size']
    rl_learning_rate = network_parameters['rl_learning_rate']
    sl_learning_rate = network_parameters['sl_learning_rate']
    memory_rl = network_parameters['memory_rl']
    memory_sl = network_parameters['memory_sl']

    expl_array  = []
    approx_expl_array = []
    ep_array = []
    game_value_array = []
    random.seed(97)

    eval_every = 5000
    #hidden_layers_sizes = [64, 64, 64]
    num_train_episodes = int(1e6)
    num_train_episodes = int(350000)
    #num_train_episodes = int(10000)
    kwargs = {
        "replay_buffer_capacity": memory_rl,
        "epsilon_decay_duration": num_train_episodes,
        "epsilon_start": 0.06,
        "epsilon_end": 0.001,
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
                  learn_every = 128,
                  optimizer_str="adam",
                  **kwargs) for idx in range(num_players)
    ]

    expl_policies_avg = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

    

    for ep in range(num_train_episodes):
        #print(ep)
        if (ep + 1) % eval_every == 0 and ep+1 > 9000:
            ep_array.append(ep)
            pi1ar_a1, pi1ar_a2, pi1ar_a3, pi2ar_a1, pi2ar_a2, pi2ar_a3, pi1ar_a1, pi1ar_a2, pi1ar_a3, pi1ar_d3, pi1ar_d2, pi1ar_d1 = create_plots(agents)

            #print("calculating approx expl..")
            #approxexpl = OptimalStoppingGameUtil.approx_exploitability(agents, env)
            #print("approx eplx = " + str(approxexpl[-1]))
            plt.figure()
            v = OptimalStoppingGameUtil.game_value_MC(agents, env)
            game_value_array.append(v)
            plt.plot(ep_array,game_value_array, label = "Average game value")
            plt.plot()
            plt.legend(bbox_to_anchor=(0, -0.12), loc='upper left')
            plt.xlabel("Episodes")
            plt.ylabel("Game value")
            plt.tight_layout()
            plt.savefig('game_value_array_weighted.png')


            losses = [agent.loss for agent in agents]
            logging.info("Losses: %s", losses)
            print("Calculating exploitability.. (Don't do this for large games!)")
            try:
                expl = exploitability.exploitability(env.game, expl_policies_avg)
            except Exception as e:
                expl = 0
                print(e)
                print("some exception when calcluation exploitability. keeps training")          
            logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
            logging.info("_____________________________________________")
            print("[%s] Exploitability AVG %s" + str(ep + 1) +" "+ str(expl))
            
            expl_array.append(expl)
            #approx_expl_array.append(approxexpl[-1])
            
            
        
        time_step = env.reset()
        #print(time_step)
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            time_step.observations["info_state"] = round_vec(time_step.observations["info_state"])
           
            # print(f"time_step.observations:{time_step.observations}")
            action_output = agents[player_id].step(time_step)
            
            s = env.get_state
            action = [action_output.action]
            time_step = env.step(action)
           
            #Update pi2
            time_step.observations["info_state"] = round_vec(time_step.observations["info_state"])
            current_l = time_step.observations["info_state"][0][0]
            current_belief = time_step.observations["info_state"][0][1]

            new_pi_2 = OptimalStoppingGameUtil.update_pi_2(agents[1],current_belief,current_l)
            s.update_pi_2(new_pi_2)
        
            
            
        # Episode is over, step all agents with final info state.
        #print("epsiode over")
        for agent in agents:
            agent.step(time_step)
    
    #abs_error = np.abs(np.subtract(approx_expl_array,expl_array))
    #plt.plot( ep_array,approx_expl_array, label = "Approx exploitability")
    plt.figure()
    plt.plot(ep_array, expl_array, label = "Exploitability")
    #plt.plot(abs_error, label = "Absolute error exploit vs approx exploit")
    plt.plot()
    plt.legend(loc="upper left")
    plt.savefig('exploit_biggame_piplotting.png')
    
    save_name = "Exploit_weighted_dist2"
    if not os.path.isfile(save_name+".csv"):
        df = pd.DataFrame()
        df2 = pd.DataFrame()
        df["exploit " ] = expl_array
        df["value " ] = game_value_array
        df2["pi1ar_d1"] = pi1ar_d1
        df2["pi1ar_d2"] = pi1ar_d2
        df2["pi1ar_d3"] = pi1ar_d3
        df2["pi1ar_a1"] = pi1ar_a1
        df2["pi1ar_a2"] = pi1ar_a2
        df2["pi1ar_a3"] = pi1ar_a3
        df2["pi2ar_a1"] = pi2ar_a1
        df2["pi2ar_a2"] = pi2ar_a2
        df2["pi2ar_a3"] = pi2ar_a3
        #df["a_expl " +(str(network_parameters))] = approx_expl_array
        df.to_csv(save_name + ".csv", header='column_names')
        df2.to_csv(save_name + "_belief.csv", header='column_names')
        
    else: # else it exists so append without writing the header
        #print("hej")
        df = pd.read_csv(save_name)
        df2 = pd.DataFrame()
        df["exploit " ] = expl_array
        df["value " ] = game_value_array
        df2["pi1ar_d1"] = pi1ar_d1
        df2["pi1ar_d2"] = pi1ar_d2
        df2["pi1ar_d3"] = pi1ar_d3
        df2["pi1ar_a1"] = pi1ar_a1
        df2["pi1ar_a2"] = pi1ar_a2
        df2["pi1ar_a3"] = pi1ar_a3
        df2["pi2ar_a1"] = pi2ar_a1
        df2["pi2ar_a2"] = pi2ar_a2
        df2["pi2ar_a3"] = pi2ar_a3
        #df["a_expl " +(str(network_parameters))] = approx_expl_array
        df.to_csv(save_name + ".csv", header='column_names')
        df2.to_csv(save_name + "_belief.csv", header='column_names')

        df.to_csv(save_name)

def create_plots(agents):
    print("Creating plots..")
    b_vec = np.arange(0,1.02,0.02)
    br_a3 = {}
    ar_a3 = {}
    br_d3 = {}
    ar_d3 = {}

    br_a2 = {}
    ar_a2 = {}
    br_d2 = {}
    ar_d2 = {}

    br_a1 = {}
    ar_a1 = {}
    br_d1 = {}
    ar_d1 = {}
    #d = {}
    for b in b_vec:
        br_a3[b] = OptimalStoppingGameUtil.update_pi_2(agents[1],b,3, temp_mode=nfsp.MODE.best_response, is_temp_mode=True)
        ar_a3[b] = OptimalStoppingGameUtil.update_pi_2(agents[1],b,3, temp_mode=nfsp.MODE.average_policy, is_temp_mode=True)

        br_d3[b] = OptimalStoppingGameUtil.get_pi_1(agents[0],b,3, temp_mode=nfsp.MODE.best_response, is_temp_mode=True)
        ar_d3[b] = OptimalStoppingGameUtil.get_pi_1(agents[0],b,3, temp_mode=nfsp.MODE.average_policy, is_temp_mode=True)

        br_a2[b] = OptimalStoppingGameUtil.update_pi_2(agents[1],b,2, temp_mode=nfsp.MODE.best_response, is_temp_mode=True)
        ar_a2[b] = OptimalStoppingGameUtil.update_pi_2(agents[1],b,2, temp_mode=nfsp.MODE.average_policy, is_temp_mode=True)

        br_d2[b] = OptimalStoppingGameUtil.get_pi_1(agents[0],b,2, temp_mode=nfsp.MODE.best_response, is_temp_mode=True)
        ar_d2[b] = OptimalStoppingGameUtil.get_pi_1(agents[0],b,2, temp_mode=nfsp.MODE.average_policy, is_temp_mode=True)

        br_a1[b] = OptimalStoppingGameUtil.update_pi_2(agents[1],b,2, temp_mode=nfsp.MODE.best_response, is_temp_mode=True)
        ar_a1[b] = OptimalStoppingGameUtil.update_pi_2(agents[1],b,2, temp_mode=nfsp.MODE.average_policy, is_temp_mode=True)

        br_d1[b] = OptimalStoppingGameUtil.get_pi_1(agents[0],b,1, temp_mode=nfsp.MODE.best_response, is_temp_mode=True)
        ar_d1[b] = OptimalStoppingGameUtil.get_pi_1(agents[0],b,1, temp_mode=nfsp.MODE.average_policy, is_temp_mode=True)

    pi1br_a3 = []
    pi2br_a3 = []
    pi1ar_a3 = []
    pi2ar_a3 = []

    pi1br_d3 = []
    pi1ar_d3 = []

    pi1br_a2 = []
    pi2br_a2 = []
    pi1ar_a2 = []
    pi2ar_a2 = []

    pi1br_d2 = []
    pi1ar_d2 = []


    pi1br_a1 = []
    pi2br_a1 = []
    pi1ar_a1 = []
    pi2ar_a1 = []

    pi1br_d1 = []
    pi1ar_d1 = []


                #d = {}
    for v in br_a3.values():
        #rint(v)
        pi1br_a3.append(v[0][0])
        pi2br_a3.append(v[1][0])

    for v in ar_a3.values():
        #rint(v)
        pi1ar_a3.append(v[0][0])
        pi2ar_a3.append(v[1][0])

    for v in br_d3.values():
        #rint(v)
        pi1br_d3.append(v[0][0])

    for v in ar_d3.values():
        #rint(v)
        pi1ar_d3.append(v[0][0]) 
    
    #L = 2

    for v in br_a2.values():
        #rint(v)
        pi1br_a2.append(v[0][0])
        pi2br_a2.append(v[1][0])

    for v in ar_a2.values():
        #rint(v)
        pi1ar_a2.append(v[0][0])
        pi2ar_a2.append(v[1][0])

    for v in br_d2.values():
        #rint(v)
        pi1br_d2.append(v[0][0])

    for v in ar_d2.values():
        #rint(v)
        pi1ar_d2.append(v[0][0]) 
    
    #L = 1

    for v in br_a1.values():
        #rint(v)
        pi1br_a1.append(v[0][0])
        pi2br_a1.append(v[1][0])

    for v in ar_a1.values():
        #rint(v)
        pi1ar_a1.append(v[0][0])
        pi2ar_a1.append(v[1][0])

    for v in br_d1.values():
        #rint(v)
        pi1br_d1.append(v[0][0])

    for v in ar_d1.values():
        #rint(v)
        pi1ar_d1.append(v[0][0]) 


    plt.figure()
    plt.plot(b_vec,pi1br_a3, label = "Attacker BR Probability of action 0 in state 0 with l=3")
    plt.plot(b_vec,pi2br_a3, label = "Attacker BR Probability of action 0 in state 1 with l=3")

    plt.plot(b_vec,pi1ar_a3, label = "Attacker AR Probability of action 0 in state 0 with l=3")
    plt.plot(b_vec,pi2ar_a3, label = "Attacker AR Probability of action 0 in state 1 with l=3")

    plt.legend(bbox_to_anchor=(0, -0.12), loc='upper left')
    plt.xlabel("Defender belief")
    plt.ylabel("Action probability")
    plt.tight_layout()
    plt.savefig('pi_attacker_plot3_biggame.png')

    plt.figure()
    plt.plot(b_vec,pi1br_a2, label = "Attacker BR Probability of action 0 in state 0 with l=2")
    plt.plot(b_vec,pi2br_a2, label = "Attacker BR Probability of action 0 in state 1 with l=2")

    plt.plot(b_vec,pi1ar_a2, label = "Attacker AR Probability of action 0 in state 0 with l=2")
    plt.plot(b_vec,pi2ar_a2, label = "Attacker AR Probability of action 0 in state 1 with l=2")

    plt.legend(bbox_to_anchor=(0, -0.12), loc='upper left')
    plt.xlabel("Defender belief")
    plt.ylabel("Action probability")
    plt.tight_layout()
    plt.savefig('pi_attacker_plot2_biggame.png')

    plt.figure()

    plt.plot(b_vec,pi1br_a1, label = "Attacker BR Probability of action 0 in state 0 with l=1")
    plt.plot(b_vec,pi2br_a1, label = "Attacker BR Probability of action 0 in state 1 with l=1")

    plt.plot(b_vec,pi1ar_a1, label = "Attacker AR Probability of action 0 in state 0 with l=1")
    plt.plot(b_vec,pi2ar_a1, label = "Attacker AR Probability of action 0 in state 1 with l=1")

    plt.legend(bbox_to_anchor=(0, -0.12), loc='upper left')
    plt.xlabel("Defender belief")
    plt.ylabel("Action probability")
    plt.tight_layout()
    plt.savefig('pi_attacker_plot1_biggame.png')

    plt.figure()
    plt.plot(b_vec,pi1br_d3, label = "Defender BR Probability of action 0 as func of belief with l=3")

    plt.plot(b_vec,pi1ar_d3, label = "Defender AR Probability of action 0 as func of belief with l=3")

    #plt.plot(abs_error, label = "Absolute error exploit vs approx exploit")
    plt.plot()
    plt.legend(bbox_to_anchor=(0, -0.12), loc='upper left')
    plt.xlabel("Defender belief")
    plt.ylabel("Action probability")
    plt.tight_layout()
    plt.savefig('pi_defender_plot3_biggame.png')

    plt.figure()
    plt.plot(b_vec,pi1br_d2, label = "Defender BR Probability of action 0 as func of belief with l=2")

    plt.plot(b_vec,pi1ar_d2, label = "Defender AR Probability of action 0 as func of belief with l=2")

    #plt.plot(abs_error, label = "Absolute error exploit vs approx exploit")
    plt.plot()
    plt.legend(bbox_to_anchor=(0, -0.12), loc='upper left')
    plt.xlabel("Defender belief")
    plt.ylabel("Action probability")
    plt.tight_layout()
    plt.savefig('pi_defender_plot2_biggame.png')
    

    plt.figure()
    plt.plot(b_vec,pi1br_d1, label = "Defender BR Probability of action 0 as func of belief with l=1")

    plt.plot(b_vec,pi1ar_d1, label = "Defender AR Probability of action 0 as func of belief with l=1")

    #plt.plot(abs_error, label = "Absolute error exploit vs approx exploit")
    plt.plot()
    plt.legend(bbox_to_anchor=(0, -0.12), loc='upper left')
    plt.xlabel("Defender belief")
    plt.ylabel("Action probability")
    plt.tight_layout()
    plt.savefig('pi_defender_plot1_biggame.png')
    
    
    print("Plots created")
    return pi1ar_a1, pi1ar_a2, pi1ar_a3, pi2ar_a1, pi2ar_a2, pi2ar_a3, pi1ar_a1, pi1ar_a2, pi1ar_a3, pi1ar_d3, pi1ar_d2, pi1ar_d1


if __name__ == "__main__":
    app.run(main)
