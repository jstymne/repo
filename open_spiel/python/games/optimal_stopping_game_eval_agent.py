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

            self.pi_2_stage = [[0.5, 0.5],[0.5, 0.5],[0.5, 0.5]]
            if self.evaluation_type == "RandomAttacker":
                self.pi_2_stage = [[1-self.random_stopping_prob, self.random_stopping_prob], [1-self.random_stopping_prob, self.random_stopping_prob], [0.5, 0.5]]
            

    def step(self,time_step, is_evaluation = True):
        if self.evaluation_type == "RandomAttacker":
            
            probs = np.zeros(2)
            probs[OptimalStoppingGameAction.CONTINUE] = self.random_stopping_prob
            probs[OptimalStoppingGameAction.STOP] = 1-self.random_stopping_prob

            action = np.random.choice(len(probs), p=probs)
            output = rl_agent.StepOutput(action=action, probs=probs)
            return output

        if self.evaluation_type == "HeuristicAttacker":
            
            
            current_belief = time_step.observations["info_state"][1][1]
            current_state = time_step.observations["info_state"][1][2]

            if current_belief > 0.5:
                self.pi_2_stage = [[0.2, 0.8],[0.8, 0.2],[0.5, 0.5]]
            else:
                self.pi_2_stage = [[0.8, 0.2],[0.2, 0.8],[0.5, 0.5]]
            
            if current_state == OptimalStoppingGameStateType.NO_INTRUSION:
                probs = np.zeros(2)
                
                if current_belief < 0.5: #Higher chance to attack            
                    probs[OptimalStoppingGameAction.CONTINUE] = 0.2
                    probs[OptimalStoppingGameAction.STOP] = 0.8

                else: #High current belief, lower chance to attack
                    probs[OptimalStoppingGameAction.CONTINUE] = 0.8
                    probs[OptimalStoppingGameAction.STOP] = 0.2

                action = np.random.choice(len(probs), p=probs)
                output = rl_agent.StepOutput(action=action, probs=probs)
                return output
            
            elif current_state == OptimalStoppingGameStateType.INTRUSION:
                probs = np.zeros(2)
                
                if current_belief < 0.5: #Low belief, higher chance to continue            
                    probs[OptimalStoppingGameAction.CONTINUE] = 0.8
                    probs[OptimalStoppingGameAction.STOP] = 0.2

                else: #High belief, higher chance to stop
                    probs[OptimalStoppingGameAction.CONTINUE] = 0.2
                    probs[OptimalStoppingGameAction.STOP] = 0.8

                action = np.random.choice(len(probs), p=probs)
                output = rl_agent.StepOutput(action=action, probs=probs)
                return output
            
            elif time_step.last():
                return
    def prep_next_episode_MC(self, time_step):
        pass