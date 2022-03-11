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

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.pytorch import nfsp
from open_spiel.python.games.optimal_stopping_game_config import OptimalStoppingGameConfig


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
    # params["R_SLA"] = 10
    game = pyspiel.load_game("python_optimal_stopping_game", params)
    num_players = game.config.num_players
    game = pyspiel.convert_to_turn_based(game)

    env = rl_environment.Environment(game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    eval_every = 10000
    hidden_layers_sizes = [64, 64, 64]
    num_train_episodes = int(3e6)
    kwargs = {
        "replay_buffer_capacity": int(2e5),
        "epsilon_decay_duration": num_train_episodes,
        "epsilon_start": 0.06,
        "epsilon_end": 0.001,
    }

    agents = [
        nfsp.NFSP(player_id=idx,
                  state_representation_size = info_state_size,
                  num_actions = num_actions,
                  hidden_layers_sizes = hidden_layers_sizes,
                  reservoir_buffer_capacity = int(2e5),
                  anticipatory_param = 0.1,
                  batch_size = 256,
                  rl_learning_rate = 0.001,
                  sl_learning_rate = 0.001,
                  min_buffer_size_to_learn = 2000,
                  learn_every = 128,
                  **kwargs) for idx in range(num_players)
    ]
    expl_policies_avg = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

    for ep in range(num_train_episodes):
        if (ep + 1) % eval_every == 0:
            losses = [agent.loss for agent in agents]
            logging.info("Losses: %s", losses)
            print("Calculating exploitability.. (Don't do this for large games!)")
            expl = exploitability.exploitability(env.game, expl_policies_avg)
            logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
            logging.info("_____________________________________________")

        time_step = env.reset()
        while not time_step.last():

            player_id = time_step.observations["current_player"]
            time_step.observations["info_state"] = round_vec(time_step.observations["info_state"])
            # print(f"time_step.observations:{time_step.observations}")
            action_output = agents[player_id].step(time_step)
            s = env.get_state
            action_list = [action_output.action]
            time_step = env.step(action_list)
            print(time_step)

        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)



if __name__ == "__main__":
    app.run(main)
