
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from open_spiel.python.games.optimal_stopping_game_config import OptimalStoppingGameConfig
from open_spiel.python.games.optimal_stopping_game_observation_type import OptimalStoppingGameObservationType
from open_spiel.python.games.optimal_stopping_game_action import OptimalStoppingGameAction
from open_spiel.python.games.optimal_stopping_game_state_type import OptimalStoppingGameStateType
from open_spiel.python import rl_agent
from typing import List


class OptimalStoppingEvalAgent(rl_agent.AbstractAgent):
  """Random agent class."""

  def __init__(self, player_id, num_actions,  evaluation_type = "RandomAttacker", random_stopping_prob = 0.5):
    assert num_actions > 0
    self._player_id = player_id
    self._num_actions = num_actions
    self.evaluation_type = evaluation_type
    self.random_stopping_prob = random_stopping_prob 

    self.pi_2_stage = [[0.5, 0.5],[0.5, 0.5],[0.5, 0.5]]
    if self.evaluation_type == "RandomAttacker":
        self.pi_2_stage = [[1-self.random_stopping_prob, self.random_stopping_prob], [1-self.random_stopping_prob, self.random_stopping_prob], [0.5, 0.5]]
  

  def prep_next_episode_MC(self, time_step):
        pass

  def step(self, time_step, is_evaluation=False):
   
   
    # If it is the end of the episode, don't select an action.
    if time_step.last():
      return
    if self.evaluation_type == "RandomAttacker":
        # Pick a random legal action.
        cur_legal_actions = time_step.observations["legal_actions"][self._player_id]
        action = np.random.choice(cur_legal_actions)
        probs = np.zeros(self._num_actions)
        probs[cur_legal_actions] = 1.0 / len(cur_legal_actions)

        return rl_agent.StepOutput(action=action, probs=probs)

    elif self.evaluation_type == "HeuristicAttacker":
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

    
class OptimalStoppingEvalAgent2:

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

        elif self.evaluation_type == "HeuristicAttacker":
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