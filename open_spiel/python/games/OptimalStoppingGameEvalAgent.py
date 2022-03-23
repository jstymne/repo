from typing import List
import numpy as np
from open_spiel.python.games.optimal_stopping_game_config import OptimalStoppingGameConfig
from open_spiel.python.games.optimal_stopping_game_observation_type import OptimalStoppingGameObservationType
from open_spiel.python.games.optimal_stopping_game_action import OptimalStoppingGameAction
from open_spiel.python.games.optimal_stopping_game_state_type import OptimalStoppingGameStateType

from open_spiel.python import rl_agent
class OptimalStoppingEvalAgent:

    def __init__(self,
                evaluation_type = "RandomAttacker",
                random_stopping_prob = 0.5
                ):
            self.evaluation_type = evaluation_type
            self.random_stopping_prob = random_stopping_prob

    def step(self,time_step):
        if self.evaluation_type == "RandomAttacker":
            probs = np.zeros(2)
            probs[OptimalStoppingGameAction.CONTINUE] = self.random_stopping_prob
            probs[OptimalStoppingGameAction.STOP] = 1-self.random_stopping_prob
            action = np.random.choice(len(probs), p=probs)
            output = rl_agent.StepOutput(action=action, probs=probs)
            return output

        if self.evaluation_type == "HeuresticAttacker":
            
            current_belief = time_step.observations["info_state"][1][1]
            current_state = time_step.observations["info_state"][1][2]


            if current_state == NO_INTRUSION
            probs = np.zeros(2)
            probs[OptimalStoppingGameAction.CONTINUE] = self.random_stopping_prob
            probs[OptimalStoppingGameAction.STOP] = 1-self.random_stopping_prob
            action = np.random.choice(len(probs), p=probs)
            output = rl_agent.StepOutput(action=action, probs=probs)
            return output

            pass #evaluate heuristic policy
        
a = OptimalStoppingEvalAgent()
b = a.step(1)
print(b)