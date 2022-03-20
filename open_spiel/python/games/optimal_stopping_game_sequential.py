import pyspiel
from open_spiel.python.games.optimal_stopping_game_config_sequential import OptimalStoppingGameConfigSequential
from open_spiel.python.games.optimal_stopping_game_state_sequential import OptimalStoppingGameStateSequential
from open_spiel.python.games.optimal_stopping_game_observer import OptimalStoppingGameObserver


class OptimalStoppingGameSequential(pyspiel.Game):
    """
    Optimal stopping game to model an intrusion prevention scenario.
    More information can be found here: https://arxiv.org/abs/2111.00289
    """

    def __init__(self, params : dict = OptimalStoppingGameConfigSequential.default_params()):
        """
        Initializes the game

        :param params: user-supplied parameters of the game
        """
        config = OptimalStoppingGameConfigSequential.from_params_dict(params_dict=params)
        super().__init__(config.create_game_type(),config.create_game_info(), params)
        self.config = config
        self.params = params

    def observation_tensor_size(self):
        """
        Method to conform with the pyspiel API

        :return: the size of the observation tensor of the game
        """
        return self.config.observation_tensor_size

    def observation_tensor_shape(self):
        """
        Method to conform with the pyspiel API

        :return: shape of the observation tensor of the game
        """
        return self.config.observation_tensor_shape

    def information_state_tensor_size(self):
        """
        Method to conform with the pyspiel API

        :return: the size of the information state tensor of the game
        """
        return self.config.information_state_tensor_size

    def information_state_tensor_shape(self):
        """
        Method to conform with the pyspiel API

        :return: the size of the information state tensor of the game
        """
        return self.config.information_state_tensor_shape

    def new_initial_state(self) -> OptimalStoppingGameStateSequential:
        """
        :return: the initial state of the game (root node in the extensive-form tree)
        """
        return OptimalStoppingGameStateSequential(self, config=self.config)

    def make_py_observer(self, iig_obs_type=None, params=None) -> OptimalStoppingGameObserver:
        """
        Method to conform with the pyspiel API

        :param iig_obs_type: the observation type
        :param params: extra parametes
        :return: an object that can be used to observe the game state
        """
        return OptimalStoppingGameObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            params)

# Register the game with the OpenSpiel library
pyspiel.register_game(
    OptimalStoppingGameConfigSequential.from_params_dict(params_dict=OptimalStoppingGameConfigSequential.default_params()).create_game_type(),
    OptimalStoppingGameSequential)
