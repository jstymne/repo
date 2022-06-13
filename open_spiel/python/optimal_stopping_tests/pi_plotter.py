import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch import bitwise_left_shift
from open_spiel.python.games.optimal_stopping_game_eval_agent import OptimalStoppingEvalAgent
from open_spiel.python.games.optimal_stopping_game_player_type import OptimalStoppingGamePlayerType
from open_spiel.python import rl_environment

######################################################### Eval agent policy plots

#plt.style.use("ggplot")

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

axs[0].set_title('Random attacker strategies', fontsize = 10)
axs[1].set_title('Heuristic attacker strategies', fontsize = 10)

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
#Exploit_new_code_base_with_good_params_leduc
#Exploit_new_code_base_with_good_params_leduc_1m
name_str = "Exploit_new_code_base_with_good_params_leduc_1m"
#name_str = "big_game_test"
number_exp = 4
number_ep = 50
window_size = 8
#Basecase
fig = plt.figure()
ax = fig.add_subplot(111)


rolling_exploits = np.zeros((number_exp,number_ep-(window_size-1)))
approx_exploits = np.zeros((number_exp,number_ep))
approx_exp_NFSPs = np.zeros((number_exp,number_ep))
new_exploit_calc = np.zeros((number_exp,number_ep))

for i in range (1,number_exp+1):
    df = pd.read_csv(name_str+ str(i) + ".csv")
    
    approx_exploit = df["approx_expl_array" ]
    approx_exploits[i-1] = approx_exploit
    
    approx_exp_NFSP = df["approx_exp_NFSP_array" ]
    approx_exp_NFSPs[i-1] = approx_exp_NFSP


    windows = approx_exp_NFSP.rolling(window_size)
    
    # Create a series of moving
    # averages of each window
    moving_averages = windows.mean()

    # Convert pandas series back to list
    moving_averages_list = moving_averages.tolist()
    
    # Remove null entries from the list
    exploit = moving_averages_list[(window_size-1):]
    rolling_exploits[i-1] = exploit
    
    

#exploit_errors = np.std(exploits,0)

approx_averages = np.average((approx_exploits),0)
approx_averages_rolling = np.average(rolling_exploits,0)
approx_errors = np.std(approx_exploits,0)

approx_NFSP_averages = np.average(abs(approx_exp_NFSPs),0)
approx_NFSP_errors = np.std(approx_exp_NFSPs,0)
episodes = []
episodes2 = []
for i in range(number_ep):
    episodes.append(20000 +i*20000)
for i in range(number_ep-(window_size-1)):
    episodes2.append(100000 + (window_size-1)*100000 +i*100000)
#print(errors)
# Make the plot
#plt.plot(episodes2, approx_averages_rolling, label= "Exploit")
#plt.errorbar(episodes, new_averages, new_averages_errors, linestyle='None', marker='^')

plt.errorbar(episodes, approx_averages, approx_errors, linestyle='None', marker='^')
#plt.plot(episodes,new_averages)
#plt.plot(episodes,approx_averages,label="Approx expl")
#plt.plot(episodes,approx_NFSP_averages,label="Approx  NFSP expl")
#plt.legend()
# Adding Xticks

plt.xlabel("Episodes", fontweight ='bold', fontsize = 16)
plt.ylabel("Approximate Exploitability", fontweight ='bold', fontsize = 16)
#plt.xticks([r + barWidth/2 for r in range(len(obs))],
 #       obs)
#ax.set_yscale("log")
plt.savefig("exploitability_smallgame_basecase")

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(episodes, approx_averages, label = "Approximate game value from defender perspective", color = '#FF9848')
ax.fill_between(episodes, approx_averages+approx_errors, approx_averages-approx_errors, alpha=0.3, facecolor='#FF9848', label = "Exploitability range for defender vs NFSP")


plt.savefig('expl_test_plot')



######################################################### One file test 

file_name = "Exploit_new_code_base_with_good_params_leduc_1m_bigdist3"
#name_str = "big_game_test"
#number_exp = 3
#number_ep = 50
window_size = 8
#Basecase
fig = plt.figure()
ax = fig.add_subplot(111)

df = pd.read_csv(file_name + ".csv")
    
approx_exploit = df["approx_expl_array" ]

plt.plot(episodes, approx_exploit, linestyle='None', marker='^')


plt.xlabel("Episodes", fontweight ='bold', fontsize = 16)
plt.ylabel("Approximate Exploitability", fontweight ='bold', fontsize = 16)
#plt.xticks([r + barWidth/2 for r in range(len(obs))],
 #       obs)
#ax.set_yscale("log")
plt.savefig("one_file_test")



######################################################### Game value plot with bound 

#Basecase


values = np.zeros((number_exp,number_ep))
br_attacker_values = np.zeros((number_exp,number_ep))
br_defender_values = np.zeros((number_exp,number_ep))
for i in range (1,number_exp+1):
    df = pd.read_csv(name_str+ str(i) + ".csv")
    
    value = df["value"]
    values[i-1] = value

    br_attacker_value = df["avg_attacker_br_R_array"]
    br_attacker_values[i-1] = br_attacker_value

    br_defender_value = df["avg_defender_br_R_array"]
    br_defender_values[i-1] = br_defender_value
    
value_averages = np.average(values,0)
value_errors = np.std(values,0)
br_attacker_value_averages = np.average(br_attacker_values,0)
br_attacker_value_errors = np.std(br_attacker_values,0)

br_defender_value_averages = np.average(br_defender_values,0)
br_defender_value_errors = np.std(br_defender_values,0)

#plt.errorbar(episodes, value_averages, value_errors, linestyle='None', marker='^', label = "Values")
#plt.errorbar(episodes, random_value_averages, random_value_errors, linestyle='None', marker='^', label = "Rand values")
#plt.errorbar(episodes, heur_value_averages, heur_value_errors, linestyle='None', marker='^', label = "Heuristic Values")


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(episodes, br_defender_value_averages, label = "BR defender average episode reward", color = "red")

ax.plot(episodes, value_averages, label = "Approximate game value from defender perspective", color = '#FF9848')
ax.fill_between(episodes, br_defender_value_averages, value_averages, alpha=0.3, facecolor='#FF9848', label = "Exploitability range for defender vs NFSP")

plt.xlabel("Episodes", fontweight ='bold', fontsize = 10)
plt.ylabel("Average episode reward", fontweight ='bold', fontsize = 10)

handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
fig.subplots_adjust(bottom=0.33)
ax.legend(loc='lower center',  fontsize = 10,
             bbox_to_anchor=(0.5, -0.6))
ax.grid('on')
plt.savefig('value_plot_with_bound_defender')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(episodes, -value_averages, label = "Approximate game value from attacker perspective", color = 'c')
#plt.plot(episodes, exploit_averages, label = "Expl Values")

ax.plot(episodes, br_attacker_value_averages, label = "BR attacker average episode reward", color = "blue")
ax.fill_between(episodes, -value_averages, br_attacker_value_averages, alpha=0.3, facecolor='c', label = "Exploitability range for attacker vs NFSP")

plt.xlabel("Episodes", fontweight ='bold', fontsize = 10)
plt.ylabel("Average episode reward", fontweight ='bold', fontsize = 10)

handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
fig.subplots_adjust(bottom=0.33)
ax.legend(loc='lower center',  fontsize = 10,
             bbox_to_anchor=(0.5, -0.6))
ax.grid('on')
plt.savefig('value_plot_with_bound_attacker')




#plt.fill_between)

# Adding Xticks


#plt.xticks([r + barWidth/2 for r in range(len(obs))],
 #       obs)
#ax = plt.gca()
#plt.legend(fontsize = 16, bbox_to_anchor=(0.67, 1.15), bbox_transform=ax.transAxes)
#plt.legend()





######################################################### Game value plots 

#Basecase
#plt.figure()
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



# Make the plot

#plt.errorbar(episodes, value_averages, value_errors, linestyle='None', marker='^', label = "Values")
#plt.errorbar(episodes, random_value_averages, random_value_errors, linestyle='None', marker='^', label = "Rand values")
#plt.errorbar(episodes, heur_value_averages, heur_value_errors, linestyle='None', marker='^', label = "Heuristic Values")



fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(x, np.sin(x), label='Sine')



ax.plot(episodes, value_averages, label = "Approximate game value", color = '#FF9848')
ax.plot(episodes, random_value_averages, label = "Average episode reward versus random attacker", color = "purple")
ax.plot(episodes, heur_value_averages, label = "Average episode reward versus heuristic attacker", color = "green")
ax.plot(episodes, -br_attacker_value_averages, label = "Average episode reward versus BR attacker ", color = "red")
#plt.plot(episodes, exploit_averages, label = "Expl Values")
#error = exploit_averages
#plt.fill_between)
#plt.fill_between(episodes, value_averages-error, value_averages+error, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label = "Exploitability range")

# Adding Xticks
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))

plt.xlabel("Episodes", fontweight ='bold', fontsize = 10)
plt.ylabel("Average episode reward", fontweight ='bold', fontsize = 10)

fig.subplots_adjust(bottom=0.33)

ax.legend(loc='lower center',  fontsize = 10,
             bbox_to_anchor=(0.5, -0.6))
ax.grid('on')
#plt.xticks([r + barWidth/2 for r in range(len(obs))],
 #       obs)
#ax = plt.gca()
#plt.tight_layout()
#plt.legend(fontsize = 10, bbox_to_anchor=(1, 0), bbox_transform=ax.transAxes)
#plt.legend(bbox_to_anchor =(0.5,-0.27), loc='lower center')
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



axs[0, 0].set_title('Attacker strategy for l = 3', fontsize = 8)
axs[0, 1].set_title('Attacker strategy for l = 2', fontsize = 8)
axs[0, 2].set_title('Attacker strategy for l = 1', fontsize = 8)

axs[1, 0].set_title('Defender strategy for l = 3', fontsize = 8)
axs[1, 1].set_title('Defender strategy for l = 2', fontsize = 8)
axs[1, 2].set_title('Defender strategy for l = 1', fontsize = 8)

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


#########################################################  Policy plots comparision biggame

first_filename = "Exploit_new_code_base_with_good_params_leduc_1m"
compare_filename = "Exploit_new_code_base_with_good_params_leduc_1m_bigdist"
number_exp = 3
plt.figure()
fig, axs = plt.subplots(2, 3)
j = 0
attacker_policies = [["attacker_stopping_probabilities_intrusion_3","attacker_stopping_probabilities_no_intrusion_3"], ["attacker_stopping_probabilities_intrusion_2","attacker_stopping_probabilities_no_intrusion_2"],["attacker_stopping_probabilities_intrusion_1","attacker_stopping_probabilities_no_intrusion_1"]] 
defender_policies = ["defender_stopping_probabilities_3", "defender_stopping_probabilities_2", "defender_stopping_probabilities_1"]
b_vec = np.linspace(0, 1, num=100)

for policy in attacker_policies:
    state0_probs_normal = np.zeros((number_exp,100))
    state1_probs_normal = np.zeros((number_exp,100))
    state0_probs_compare = np.zeros((number_exp,100))
    state1_probs_compare = np.zeros((number_exp,100))
    
    for i in range (1,number_exp+1):
        df = pd.read_csv(first_filename+ str(i) + "_belief.csv")
        
        stopping_nointrusion_prob_normal = df[policy[1]]
        state0_probs_normal[i-1] = stopping_nointrusion_prob_normal
        stopping_intrusion_prob_normal = df[policy[0]]
        state1_probs_normal[i-1] = stopping_intrusion_prob_normal

        df = pd.read_csv(compare_filename+ str(i) + "_belief.csv")
        
        stopping_nointrusion_prob = df[policy[1]]
        state0_probs_compare[i-1] = stopping_nointrusion_prob
        stopping_intrusion_prob = df[policy[0]]
        state1_probs_compare[i-1] = stopping_intrusion_prob

    
    state0_averages_attacker = np.average(state0_probs_normal,0)
    state0_errors_attacker = np.std(state0_probs_normal,0)
    state1_averages_attacker = np.average(state1_probs_normal,0)
    state1_errors_attacker = np.std(state1_probs_normal,0)
    
    state0_averages_attacker_compare = np.average(state0_probs_compare,0)
    state0_errors_attacker_compare = np.std(state0_probs_compare,0)
    state1_averages_attacker_compare = np.average(state1_probs_compare,0)
    state1_errors_attacker_compare = np.std(state1_probs_compare,0)



    #axs[0, j].errorbar(b_vec, state0_averages_attacker, state0_errors_attacker, label = "Attacker stopping probability in state 0")
    #axs[0, j].errorbar(b_vec, state1_averages_attacker, state1_errors_attacker, label = "Attacker stopping probability in state 1")
    axs[0, j].plot(b_vec, state0_averages_attacker, label = "Attacker stopping probability in state 0 in limited game", color = "blue")
    axs[0, j].plot(b_vec, state0_averages_attacker_compare, label = "Attacker stopping probability in state 0 using datatrace", color = "c")
    axs[0, j].plot(b_vec, state1_averages_attacker, label = "Attacker stopping probability in state 1 in limited game", color = "orange")
    axs[0, j].plot(b_vec, state1_averages_attacker_compare, label = "Attacker stopping probability in state 1 using datatrace", color = "red")
    j = j+1
j = 0
for policy in defender_policies:
    #print(policy)
    state_probs = np.zeros((number_exp,100))
    state_probs_compare = np.zeros((number_exp,100))
    
    for i in range (1,number_exp+1):
        df = pd.read_csv(first_filename+ str(i) + "_belief.csv")
        
        prob = df[policy]
        state_probs[i-1] = prob

        df = pd.read_csv(compare_filename+ str(i) + "_belief.csv")
        
        prob = df[policy]
        state_probs_compare[i-1] = prob
    
    state_averages_defender = np.average(state_probs,0)
    state_errors_defender = np.std(state_probs,0)
    state_averages_defender_compare = np.average(state_probs_compare,0)
    state_errors_defender_compare = np.std(state_probs_compare,0)
    #print(state_averages_defender)

    #axs[1, j].errorbar(b_vec, state_averages_defender, state_errors_defender, label = "Defender stopping probability independent of state", color='green')
    axs[1, j].plot(b_vec, state_averages_defender, label = "Defender stopping probability independent of state in limited game", color='green')
    axs[1, j].plot(b_vec, state_averages_defender_compare, label = "Defender stopping probability independent of state using datatrace", color='lime')


    j = j+1



axs[0, 0].set_title('Attacker strategy for l = 3', fontsize = 8)
axs[0, 1].set_title('Attacker strategy for l = 2', fontsize = 8)
axs[0, 2].set_title('Attacker strategy for l = 1', fontsize = 8)

axs[1, 0].set_title('Defender strategy for l = 3', fontsize = 8)
axs[1, 1].set_title('Defender strategy for l = 2', fontsize = 8)
axs[1, 2].set_title('Defender strategy for l = 1', fontsize = 8)

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
fig.subplots_adjust(bottom=0.35, wspace=0.33, hspace=0.4)

ax.legend(lines , labels,loc='upper center',  fontsize = 9,
             bbox_to_anchor=(-1, -0.4))
plt.savefig("compare_with_datatrace")


#########################################################  Policy plots comparision r_int

first_filename = "Exploit_new_code_base_with_good_params_leduc_1m"
compare_filename = "Exploit_new_code_base_with_good_params_leduc_1m_morerint"
number_exp = 4
plt.figure()
fig, axs = plt.subplots(2, 3)
j = 0
attacker_policies = [["attacker_stopping_probabilities_intrusion_3","attacker_stopping_probabilities_no_intrusion_3"], ["attacker_stopping_probabilities_intrusion_2","attacker_stopping_probabilities_no_intrusion_2"],["attacker_stopping_probabilities_intrusion_1","attacker_stopping_probabilities_no_intrusion_1"]] 
defender_policies = ["defender_stopping_probabilities_3", "defender_stopping_probabilities_2", "defender_stopping_probabilities_1"]
b_vec = np.linspace(0, 1, num=100)

for policy in attacker_policies:
    state0_probs_normal = np.zeros((number_exp,100))
    state1_probs_normal = np.zeros((number_exp,100))
    state0_probs_compare = np.zeros((number_exp,100))
    state1_probs_compare = np.zeros((number_exp,100))
    
    for i in range (1,number_exp+1):
        df = pd.read_csv(first_filename+ str(i) + "_belief.csv")
        
        stopping_nointrusion_prob_normal = df[policy[1]]
        state0_probs_normal[i-1] = stopping_nointrusion_prob_normal
        stopping_intrusion_prob_normal = df[policy[0]]
        state1_probs_normal[i-1] = stopping_intrusion_prob_normal

        df = pd.read_csv(compare_filename+ str(i) + "_belief.csv")
        
        stopping_nointrusion_prob = df[policy[1]]
        state0_probs_compare[i-1] = stopping_nointrusion_prob
        stopping_intrusion_prob = df[policy[0]]
        state1_probs_compare[i-1] = stopping_intrusion_prob

    
    state0_averages_attacker = np.average(state0_probs_normal,0)
    state0_errors_attacker = np.std(state0_probs_normal,0)
    state1_averages_attacker = np.average(state1_probs_normal,0)
    state1_errors_attacker = np.std(state1_probs_normal,0)
    
    state0_averages_attacker_compare = np.average(state0_probs_compare,0)
    state0_errors_attacker_compare = np.std(state0_probs_compare,0)
    state1_averages_attacker_compare = np.average(state1_probs_compare,0)
    state1_errors_attacker_compare = np.std(state1_probs_compare,0)



    #axs[0, j].errorbar(b_vec, state0_averages_attacker, state0_errors_attacker, label = "Attacker stopping probability in state 0")
    #axs[0, j].errorbar(b_vec, state1_averages_attacker, state1_errors_attacker, label = "Attacker stopping probability in state 1")
    axs[0, j].plot(b_vec, state0_averages_attacker, label = "Attacker stopping probability in state 0 in basecase", color = "blue")
    axs[0, j].plot(b_vec, state0_averages_attacker_compare, label = "Attacker stopping probability in state 0 with R_int = -5", color = "c")
    axs[0, j].plot(b_vec, state1_averages_attacker, label = "Attacker stopping probability in state 1 in basecase", color = "orange")
    axs[0, j].plot(b_vec, state1_averages_attacker_compare, label = "Attacker stopping probability in state 1 with R_int = -5", color = "red")
    j = j+1
j = 0
for policy in defender_policies:
    #print(policy)
    state_probs = np.zeros((number_exp,100))
    state_probs_compare = np.zeros((number_exp,100))
    
    for i in range (1,number_exp+1):
        df = pd.read_csv(first_filename+ str(i) + "_belief.csv")
        
        prob = df[policy]
        state_probs[i-1] = prob

        df = pd.read_csv(compare_filename+ str(i) + "_belief.csv")
        
        prob = df[policy]
        state_probs_compare[i-1] = prob
    
    state_averages_defender = np.average(state_probs,0)
    state_errors_defender = np.std(state_probs,0)
    state_averages_defender_compare = np.average(state_probs_compare,0)
    state_errors_defender_compare = np.std(state_probs_compare,0)
    #print(state_averages_defender)

    #axs[1, j].errorbar(b_vec, state_averages_defender, state_errors_defender, label = "Defender stopping probability independent of state", color='green')
    axs[1, j].plot(b_vec, state_averages_defender, label = "Defender stopping probability independent of state in basecase", color='green')
    axs[1, j].plot(b_vec, state_averages_defender_compare, label = "Defender stopping probability independent of state with R_int = -5", color='lime')


    j = j+1



axs[0, 0].set_title('Attacker strategy for l = 3', fontsize = 8)
axs[0, 1].set_title('Attacker strategy for l = 2', fontsize = 8)
axs[0, 2].set_title('Attacker strategy for l = 1', fontsize = 8)

axs[1, 0].set_title('Defender strategy for l = 3', fontsize = 8)
axs[1, 1].set_title('Defender strategy for l = 2', fontsize = 8)
axs[1, 2].set_title('Defender strategy for l = 1', fontsize = 8)

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
fig.subplots_adjust(bottom=0.35, wspace=0.33, hspace=0.4)

ax.legend(lines , labels,loc='upper center',  fontsize = 9,
             bbox_to_anchor=(-1, -0.4))
plt.savefig("compare_with_rint")



"""
base_filename = "Exploit_new_code_good_params_approx_biggame_diminlr"

eval_eps= [999999, 1999999, 2999999, 3999999, 4999999]


plt.figure()
fig, axs = plt.subplots(2, 3)
j = 0
attacker_policies = [["attacker_stopping_probabilities_intrusion_3","attacker_stopping_probabilities_no_intrusion_3"], ["attacker_stopping_probabilities_intrusion_2","attacker_stopping_probabilities_no_intrusion_2"],["attacker_stopping_probabilities_intrusion_1","attacker_stopping_probabilities_no_intrusion_1"]] 
defender_policies = ["defender_stopping_probabilities_3", "defender_stopping_probabilities_2", "defender_stopping_probabilities_1"]
b_vec = np.linspace(0, 1, num=100)

for policy in attacker_policies:
    state0_probs_normal = np.zeros((number_exp,100))
    state1_probs_normal = np.zeros((number_exp,100))

    for eps in eval_eps:
        for i in range (1,number_exp+1):
            df = pd.read_csv(base_filename+ str(eps) + "_" + str(i) + "_belief.csv")
            
            stopping_nointrusion_prob_normal = df[policy[1]]
            state0_probs_normal[i-1] = stopping_nointrusion_prob_normal
            stopping_intrusion_prob_normal = df[policy[0]]
            state1_probs_normal[i-1] = stopping_intrusion_prob_normal


        
        state0_averages_attacker = np.average(state0_probs_normal,0)
        state0_errors_attacker = np.std(state0_probs_normal,0)
        state1_averages_attacker = np.average(state1_probs_normal,0)
        state1_errors_attacker = np.std(state1_probs_normal,0)
        
 
        #axs[0, j].errorbar(b_vec, state0_averages_attacker, state0_errors_attacker, label = "Attacker stopping probability in state 0")
        #axs[0, j].errorbar(b_vec, state1_averages_attacker, state1_errors_attacker, label = "Attacker stopping probability in state 1")
        axs[0, j].plot(b_vec, state0_averages_attacker, label = "Attacker stopping probability in state 0 for ep " + str(eps+1))
        axs[0, j].plot(b_vec, state1_averages_attacker, label = "Attacker stopping probability in state 1 for ep " + str(eps+1))
      
    j = j+1
j = 0
for policy in defender_policies:
    #print(policy)
    state_probs = np.zeros((number_exp,100))
    state_probs_compare = np.zeros((number_exp,100))
    
    for i in range (1,number_exp+1):
        df = pd.read_csv(name_str+ str(i) + "_belief.csv")
        
        prob = df[policy]
        state_probs[i-1] = prob

        df = pd.read_csv(compare_filename+ str(i) + "_belief.csv")
        
        prob = df[policy]
        state_probs_compare[i-1] = prob
    
    state_averages_defender = np.average(state_probs,0)
    state_errors_defender = np.std(state_probs,0)
    state_averages_defender_compare = np.average(state_probs_compare,0)
    state_errors_defender_compare = np.std(state_probs_compare,0)
    #print(state_averages_defender)

    #axs[1, j].errorbar(b_vec, state_averages_defender, state_errors_defender, label = "Defender stopping probability independent of state", color='green')
    axs[1, j].plot(b_vec, state_averages_defender, label = "Defender stopping probability independent of state", color='green')
    axs[1, j].plot(b_vec, state_averages_defender_compare, label = "Defender stopping probability independent of state, R_int = -5", color='red')


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
fig.subplots_adjust(bottom=0.35, wspace=0.33, hspace=0.4)

ax.legend(lines , labels,loc='upper center',  fontsize = 9,
             bbox_to_anchor=(-1, -0.4))
plt.savefig("development_of_policies")

"""
print("Graphs created")