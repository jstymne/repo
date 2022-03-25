import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch import bitwise_left_shift
from open_spiel.python.games.optimal_stopping_game_eval_agent import OptimalStoppingEvalAgent
from open_spiel.python.games.optimal_stopping_game_player_type import OptimalStoppingGamePlayerType
from open_spiel.python import rl_environment

######################################################### Eval agent policy plots


fig, axs = plt.subplots(1, 2)

belief_space = np.linspace(0, 1, num=100)
random_attacker_stopping_probabilities_no_intrusion = []
random_attacker_stopping_probabilities_intrusion = []
heur_attacker_stopping_probabilities_no_intrusion = []
heur_attacker_stopping_probabilities_intrusion = []

random_agent = OptimalStoppingEvalAgent(player_id = 1, num_actions = 2, evaluation_type = "RandomAttacker")
heur_agent = OptimalStoppingEvalAgent(player_id = 1, num_actions = 2, evaluation_type = "HeuristicAttacker")

temp_ts_intrusion = {}
temp_ts_no_intrusion = {}
l = 3
for b in belief_space:
    o = {
        'info_state': [[l, b, b],
                        [l, b, 0]],
        'legal_actions': [[],[0, 1]],
        'current_player': OptimalStoppingGamePlayerType.ATTACKER,
        "serialized_state": []
    }
    temp_ts_no_intrusion= rl_environment.TimeStep(
        observations= o, rewards=None, discounts=None, step_type=None)

    random_attacker_stopping_probabilities_no_intrusion.append(random_agent.step(temp_ts_no_intrusion).probs[1]+0.005)
    heur_attacker_stopping_probabilities_no_intrusion.append(heur_agent.step(temp_ts_no_intrusion).probs[1])

    temp_ts_intrusion = temp_ts_no_intrusion
    temp_ts_intrusion.observations["info_state"] = [[l, b, b], [l, b, 1]]
    random_attacker_stopping_probabilities_intrusion.append(random_agent.step(temp_ts_intrusion).probs[1])
    heur_attacker_stopping_probabilities_intrusion.append(heur_agent.step(temp_ts_intrusion).probs[1])
    

axs[0].plot(belief_space, random_attacker_stopping_probabilities_no_intrusion, label = "Attacker stopping probability in state 0", color = "blue")
axs[0].plot(belief_space, random_attacker_stopping_probabilities_intrusion, label = "Attacker stopping probability in state 1", color = "orange")


axs[1].plot(belief_space, heur_attacker_stopping_probabilities_no_intrusion, label = "Attacker stopping probability in state 0", color = "blue")
axs[1].plot(belief_space, heur_attacker_stopping_probabilities_intrusion, label = "Attacker stopping probability in state 1", color = "orange")

axs[0].set_title('Random attacker policies', fontsize = 10)
axs[1].set_title('Heuristic attacker policies', fontsize = 10)

for ax in axs.flat:
    ax.set(ylabel='Stopping probability', xlabel='Defender belief')

plt.setp(axs, xticks=[0.0, 0.5, 1.0],
       yticks=[0,  0.5, 1])


for ax in axs.flat:
    ax.label_outer()
#fig.tight_layout()
fig.subplots_adjust(bottom=0.3, wspace=0.33, hspace=0.4)

ax.legend(loc='upper center',  fontsize = 10,
             bbox_to_anchor=(-0.2, -0.2))

plt.savefig("eval_agent_policies")



#########################################################  Distribution plots



plt.figure()

obs = [0,1,2,3,4,5,6,7,8,9]
obs_dist = [4/20,4/20,4/20,2/20,1/20,1/20,1/20,1/20,1/20,1/20]

obs_dist_intrusion = [1/20,1/20,1/20,1/20,1/20,1/20,2/20,4/20,4/20,4/20]

plt.figure()
# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

# Set position of bar on X axis
br = np.arange(len(obs))
br1 = [x for x in br]
br2 = [x + barWidth for x in br]

# Make the plot
plt.bar(br1, obs_dist, width = barWidth,
        edgecolor ='grey',  label = "Observation probability in state 0 = no intrusion")
plt.bar(br2, obs_dist_intrusion, width = barWidth,
        edgecolor ='grey', label = "Observation probability in state 1 = intrusion")
 
# Adding Xticks

plt.xlabel("Observation", fontweight ='bold', fontsize = 16)
plt.ylabel("Observation probability", fontweight ='bold', fontsize = 16)
plt.xticks([r + barWidth/2 for r in range(len(obs))],
        obs)

ax = plt.gca()
plt.legend(fontsize = 16, bbox_to_anchor=(0.67, 1.15), bbox_transform=ax.transAxes)
plt.savefig('obs_dist_plot')




# Set position of bar on X axis

######################################################### Exploitability plots 

name_str = "Exploit_new_code_base_with_good_params"
#name_str = "big_game_test"
number_exp = 2
number_ep = 500

#Basecase
plt.figure()
exploits = np.zeros((number_exp,number_ep))
approx_exploits = np.zeros((number_exp,number_ep))
for i in range (1,number_exp+1):
    df = pd.read_csv(name_str+ str(i) + ".csv")
    exploit = df["exploit " ]
    exploits[i-1] = exploit
    #approx_exploit = df["approx_expl_array" ]
    #approx_exploits[i-1] = approx_exploit
    
exploit_averages = np.average(exploits,0)
exploit_errors = np.std(exploits,0)
#approx_averages = np.average(abs(approx_exploits),0)
#approx_errors = np.std(approx_exploits,0)
episodes = []
for i in range(len(exploit_averages)):
    episodes.append(10000 +i*10000)
#print(errors)
# Make the plot
plt.plot(episodes,exploit_averages)
#plt.errorbar(episodes, approx_averages, approx_errors, linestyle='None', marker='^')
#plt.plot(episodes,approx_averages)

# Adding Xticks

plt.xlabel("Episodes", fontweight ='bold', fontsize = 16)
plt.ylabel("Exploitability", fontweight ='bold', fontsize = 16)
#plt.xticks([r + barWidth/2 for r in range(len(obs))],
 #       obs)
plt.savefig("exploitability_smallgame_basecase")

"""
#Weird distribution
plt.figure()
exploits = np.zeros((4,29))
for i in range (1,5):
    df = pd.read_csv("Exploit_weird_dist"+ str(i) + ".csv")
    exploit = df["exploit " ]
    exploits[i-1] = exploit
    
averages = np.average(exploits,0)
errors = np.std(exploits,0)
episodes = []
for i in range(len(averages)):
    episodes.append(10000 +i*5000)
#print(errors)
# Make the plot

plt.errorbar(episodes, averages, errors, linestyle='None', marker='^')

# Adding Xticks

plt.xlabel("Episodes", fontweight ='bold', fontsize = 16)
plt.ylabel("Exploitability", fontweight ='bold', fontsize = 16)
#plt.xticks([r + barWidth/2 for r in range(len(obs))],
 #       obs)
plt.savefig("exploitability_smallgame_weirddist")


#R_SLA = 1 
plt.figure()
exploits = np.zeros((4,29))
for i in range (1,5):
    df = pd.read_csv("Exploit_rsla_dist"+ str(i) + ".csv")
    exploit = df["exploit " ]
    exploits[i-1] = exploit
    
averages = np.average(exploits,0)
errors = np.std(exploits,0)
episodes = []
for i in range(len(averages)):
    episodes.append(10000 +i*5000)
#print(errors)
# Make the plot

plt.errorbar(episodes, averages, errors, linestyle='None', marker='^')

# Adding Xticks

plt.xlabel("Episodes", fontweight ='bold', fontsize = 16)
plt.ylabel("Exploitability", fontweight ='bold', fontsize = 16)
#plt.xticks([r + barWidth/2 for r in range(len(obs))],
 #       obs)
plt.savefig("exploitability_smallgame_rsla")
"""
######################################################### Game value plots 

#Basecase
plt.figure()
values = np.zeros((number_exp,number_ep))
rand_values = np.zeros((number_exp,number_ep))
heur_values = np.zeros((number_exp,number_ep))
for i in range (1,number_exp+1):
    df = pd.read_csv(name_str+ str(i) + ".csv")
    value = df["value"]
    game_value_array_random = df["game_value_array_random"]
    game_value_array_heur = df["game_value_array_heur"]

    values[i-1] = value
    rand_values[i-1] = game_value_array_random
    heur_values[i-1] = game_value_array_heur
    
value_averages = np.average(values,0)
value_errors = np.std(values,0)

random_value_averages = np.average(rand_values,0)
random_value_errors = np.std(rand_values,0)

heur_value_averages = np.average(heur_values,0)
heur_value_errors = np.std(heur_values,0)


episodes = []
for i in range(len(exploit_averages)):
    episodes.append(10000 +i*10000)
#print(errors)
# Make the plot

#plt.errorbar(episodes, value_averages, value_errors, linestyle='None', marker='^', label = "Values")
#plt.errorbar(episodes, random_value_averages, random_value_errors, linestyle='None', marker='^', label = "Rand values")
#plt.errorbar(episodes, heur_value_averages, heur_value_errors, linestyle='None', marker='^', label = "Heuristic Values")

plt.plot(episodes, value_averages, label = "Game Values")
plt.plot(episodes, random_value_averages, label = "Random values")
plt.plot(episodes, heur_value_averages, label = "Heuristic Values")


# Adding Xticks

plt.xlabel("Episodes", fontweight ='bold', fontsize = 16)
plt.ylabel("Game value", fontweight ='bold', fontsize = 16)
#plt.xticks([r + barWidth/2 for r in range(len(obs))],
 #       obs)
ax = plt.gca()
plt.legend(fontsize = 16, bbox_to_anchor=(0.67, 1.15), bbox_transform=ax.transAxes)
plt.legend()
plt.savefig('value_plot')


######################################################### All Policy plots for basecase



plt.figure()
fig, axs = plt.subplots(2, 3)
j = 0
attacker_policies = [["attacker_stopping_probabilities_intrusion_3","attacker_stopping_probabilities_no_intrusion_3"], ["attacker_stopping_probabilities_intrusion_2","attacker_stopping_probabilities_no_intrusion_2"],["attacker_stopping_probabilities_intrusion_1","attacker_stopping_probabilities_no_intrusion_1"]] 
defender_policies = ["defender_stopping_probabilities_3", "defender_stopping_probabilities_2", "defender_stopping_probabilities_1"]
b_vec = np.linspace(0, 1, num=100)
for policy in attacker_policies:
    state0_probs = np.zeros((number_exp,100))
    state1_probs = np.zeros((number_exp,100))
    
    for i in range (1,number_exp+1):
        df = pd.read_csv(name_str+ str(i) + "_belief.csv")
        
        stopping_nointrusion_prob = df[policy[1]]
        state0_probs[i-1] = stopping_nointrusion_prob
        stopping_intrusion_prob = df[policy[0]]
        state1_probs[i-1] = stopping_intrusion_prob
    
    state0_averages_attacker = np.average(state0_probs,0)
    state0_errors_attacker = np.std(state0_probs,0)
    state1_averages_attacker = np.average(state1_probs,0)
    state1_errors_attacker = np.std(state1_probs,0)
    
    #axs[0, j].errorbar(b_vec, state0_averages_attacker, state0_errors_attacker, label = "Attacker stopping probability in state 0")
    #axs[0, j].errorbar(b_vec, state1_averages_attacker, state1_errors_attacker, label = "Attacker stopping probability in state 1")
    axs[0, j].plot(b_vec, state0_averages_attacker, label = "Attacker stopping probability in state 0", color = "blue")
    axs[0, j].plot(b_vec, state1_averages_attacker, label = "Attacker stopping probability in state 1", color = "orange")
    j = j+1
j = 0
for policy in defender_policies:
    #print(policy)
    state_probs = np.zeros((number_exp,100))
    
    for i in range (1,number_exp+1):
        df = pd.read_csv(name_str+ str(i) + "_belief.csv")
        
        prob = df[policy]
        state_probs[i-1] = prob
    
    state_averages_defender = np.average(state_probs,0)
    state_errors_defender = np.std(state_probs,0)
    #print(state_averages_defender)

    #axs[1, j].errorbar(b_vec, state_averages_defender, state_errors_defender, label = "Defender stopping probability independent of state", color='green')
    axs[1, j].plot(b_vec, state_averages_defender, label = "Defender stopping probability independent of state", color='green')

    j = j+1



axs[0, 0].set_title('Attacker policy for l = 3', fontsize = 10)
axs[0, 1].set_title('Attacker policy for l = 2', fontsize = 10)
axs[0, 2].set_title('Attacker policy for l = 1', fontsize = 10)

axs[1, 0].set_title('Defender policy for l = 3', fontsize = 10)
axs[1, 1].set_title('Defender policy for l = 2', fontsize = 10)
axs[1, 2].set_title('Defender policy for l = 1', fontsize = 10)

for ax in axs.flat:
    ax.set(ylabel='Stopping probability', xlabel='Defender belief')

plt.setp(axs, xticks=[0.0, 0.5, 1.0],
       yticks=[0,  0.5, 1])

#handles, labels = ax.get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper center')
lines_labels = [fig.axes[0].get_legend_handles_labels(), fig.axes[4].get_legend_handles_labels() ]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

#fig.legend(lines, labels, fontsize = 10, bbox_to_anchor=(0.67, 2.5), bbox_transform=ax.transAxes)
#plt.legend(fontsize = 16, )
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
#fig.tight_layout()
fig.subplots_adjust(bottom=0.3, wspace=0.33, hspace=0.4)

ax.legend(lines , labels,loc='upper center',  fontsize = 10,
             bbox_to_anchor=(-1, -0.4))
plt.savefig("basecase_average_policies")


######################################################### All Policy plots for basecase
"""

# Set position of bar on X axis

state0_probs = np.zeros((4,51))
state1_probs = np.zeros((4,51))
for i in range (1,5):
    df = pd.read_csv("Exploit_weird_dist"+ str(i) + "_belief.csv")
    
    prob = df["pi1ar_a3" ]
    state0_probs[i-1] = 1-prob

    prob = df["pi2ar_a3" ]
    state1_probs[i-1] = 1-prob
    
state0_averages_attacker = np.average(state0_probs,0)
state0_errors_attacker = np.std(state0_probs,0)

state1_averages_attacker = np.average(state1_probs,0)
state1_errors_attacker = np.std(state1_probs,0)

b_vec = np.arange(0,1.02,0.02)
#print(errors)
# Make the plot

plt.errorbar(b_vec, state0_averages_attacker, state0_errors_attacker, linestyle='None', marker='^', label = "Attacker Stopping probability in state 0")
plt.errorbar(b_vec, state1_averages_attacker, state1_errors_attacker, linestyle='None', marker='^', label = "Attacker Stopping probability in state 1")
# Adding Xticks

plt.xlabel("Defender belief", fontweight ='bold', fontsize = 16)
plt.ylabel("Stopping probability", fontweight ='bold', fontsize = 16)
#plt.legend(fontsize = 16, bbox_to_anchor=(0.67, 1.15), bbox_transform=ax.transAxes)
plt.legend(fontsize = 16)


#plt.xticks([r + barWidth/2 for r in range(len(obs))],
 #       obs)
plt.savefig("errortest2")


#ax = plt.gca()
#plt.savefig('obs_dist_plot')
"""
print("Graphs created")