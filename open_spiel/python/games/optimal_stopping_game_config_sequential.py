from typing import List
import numpy as np
import pyspiel
from open_spiel.python.games.optimal_stopping_game_action import OptimalStoppingGameAction
from open_spiel.python.games.optimal_stopping_game_state_type import OptimalStoppingGameStateType
from open_spiel.python.games.optimal_stopping_game_config_base import OptimalStoppingGameConfigBase


class OptimalStoppingGameConfigSequential(OptimalStoppingGameConfigBase):

    def __init__(self, p: float = 0.001, T_max: int = 5, L: int = 3, R_ST: int = 100, R_SLA: int = 10,
                 R_COST: int = -50, R_INT: int = -100, obs: str = "",
                 obs_dist: str = "", obs_dist_intrusion: str = "", initial_belief: str = "", use_beliefs: bool = False):
        """
        DTO class representing the configuration of the optimal stopping game
        :param p: the probability that the attacker is detected at any time-step
        :param T_max: the maximum length of the game (could be infinite)
        :param L: the number of stop actions of the defender
        :param R_ST: constant for defining the reward function
        :param R_SLA: constant for defining the reward function
        :param R_COST: constant for defining the reward function
        :param R_INT: constant for defining the reward function
        :param obs: the list of observations
        :param obs_dist_intrusion: the observation distribution
        :param initial_belief: the initial belief
        :param use_beliefs: boolean flag whether to use beliefs or not. If this is false, use observations instead.
        """
        super(OptimalStoppingGameConfigSequential, self).__init__(
            p=p,T_max=T_max,L=L,R_ST=R_ST,R_SLA=R_SLA,R_COST=R_COST,R_INT=R_INT, obs=obs, obs_dist=obs_dist,
            obs_dist_intrusion=obs_dist_intrusion, initial_belief=initial_belief, use_beliefs=use_beliefs)


    def create_game_type(self) -> pyspiel.GameType:
        """
        :return: GameType object
        """
        return pyspiel.GameType(
            short_name="python_optimal_stopping_game_sequential",
            long_name="Python Optimal Stopping Game Sequential",
            dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
            chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
            information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
            utility=pyspiel.GameType.Utility.ZERO_SUM,
            reward_model=pyspiel.GameType.RewardModel.REWARDS,
            max_num_players=self.num_players,
            min_num_players=self.num_players,
            provides_information_state_string=True,
            provides_information_state_tensor=False,
            provides_observation_string=True,
            provides_observation_tensor=True,
            provides_factored_observation_string=True,
            parameter_specification=self.params)

    def create_game_info(self) -> pyspiel.GameInfo:
        """
        :return: GameInfo object
        """
        return pyspiel.GameInfo(
            num_distinct_actions=len(self.get_actions()),
            max_chance_outcomes=len(self.obs) + 1,
            num_players=self.num_players,
            min_utility=self.R_INT*10,
            max_utility=self.R_ST*10,
            utility_sum=0.0,
            max_game_length=self.T_max)


    @staticmethod
    def from_params_dict(params_dict: dict) -> "OptimalStoppingGameConfigSequential":
        """
        Creates a config object from a user-supplied dict with parameters
        :param params_dict: the dict with parameters
        :return: a config object corresponding to the parameters in the dict
        """
        return OptimalStoppingGameConfigSequential(
            p=params_dict["p"], T_max=params_dict["T_max"], L=params_dict["L"], R_ST=params_dict["R_ST"],
            R_SLA=params_dict["R_SLA"], R_COST=params_dict["R_COST"], R_INT=params_dict["R_INT"],
            obs=params_dict["obs"],
            obs_dist_intrusion=params_dict["obs_dist_intrusion"],
            obs_dist=params_dict["obs_dist"], initial_belief=params_dict["initial_belief"],
            use_beliefs=params_dict["use_beliefs"]
        )