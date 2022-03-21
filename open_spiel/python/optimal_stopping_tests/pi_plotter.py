import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


#Basecase
plt.figure()
exploits = np.zeros((3,50))
for i in range (1,4):
    df = pd.read_csv("Exploit_new_code"+ str(i) + ".csv")
    exploit = df["exploit " ]
    exploits[i-1] = exploit
    
averages = np.average(exploits,0)
errors = np.std(exploits,0)
episodes = []
for i in range(len(averages)):
    episodes.append(10000 +i*10000)
#print(errors)
# Make the plot

plt.errorbar(episodes, averages, errors, linestyle='None', marker='^')

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


""""
TODO
"""

######################################################### All Policy plots for basecase



plt.figure()
fig, axs = plt.subplots(2, 3)
j = 0
attacker_policies = [["attacker_stopping_probabilities_intrusion_3","attacker_stopping_probabilities_no_intrusion_3"], ["attacker_stopping_probabilities_intrusion_2","attacker_stopping_probabilities_no_intrusion_2"],["attacker_stopping_probabilities_intrusion_1","attacker_stopping_probabilities_no_intrusion_1"]] 
defender_policies = ["defender_stopping_probabilities_3", "defender_stopping_probabilities_2", "defender_stopping_probabilities_1"]
b_vec = np.linspace(0, 1, num=100)
for policy in attacker_policies:
    state0_probs = np.zeros((3,100))
    state1_probs = np.zeros((3,100))
    
    for i in range (1,4):
        df = pd.read_csv("Exploit_new_code"+ str(i) + "_belief.csv")
        
        prob = df[policy[0]]
        state0_probs[i-1] = prob
        prob = df[policy[1]]
        state1_probs[i-1] = prob
    
    state0_averages_attacker = np.average(state0_probs,0)
    state0_errors_attacker = np.std(state0_probs,0)
    state1_averages_attacker = np.average(state1_probs,0)
    state1_errors_attacker = np.std(state1_probs,0)
    
    axs[0, j].errorbar(b_vec, state0_averages_attacker, state0_errors_attacker, label = "Attacker stopping probability in state 0")
    axs[0, j].errorbar(b_vec, state1_averages_attacker, state1_errors_attacker, linestyle='None', marker='^', label = "Attacker stopping probability in state 1")
    j = j+1
j = 0
for policy in defender_policies:
    print(policy)
    state_probs = np.zeros((3,100))
    
    for i in range (1,4):
        df = pd.read_csv("Exploit_new_code"+ str(i) + "_belief.csv")
        
        prob = df[policy]
        state_probs[i-1] = prob
    
    state_averages_defender = np.average(state_probs,0)
    state_errors_defender = np.std(state_probs,0)
    print(state_averages_defender)

    axs[1, j].errorbar(b_vec, state_averages_defender, state_errors_defender, linestyle='None', marker='^', label = "Defender stopping probability independent of state", color='green')
    j = j+1



axs[0, 0].set_title('Attacker policy for l = 1')
axs[0, 1].set_title('Attacker policy for l = 2')
axs[0, 2].set_title('Attacker policy for l = 3')

axs[1, 0].set_title('Defender policy for l = 1')
axs[1, 1].set_title('Defender policy for l = 2')
axs[1, 2].set_title('Defender policy for l = 3')

for ax in axs.flat:
    ax.set(xlabel='Stopping probability', ylabel='Defender belief')

#handles, labels = ax.get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper center')
lines_labels = [fig.axes[0].get_legend_handles_labels(), fig.axes[4].get_legend_handles_labels() ]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

fig.legend(lines, labels)
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.savefig("basecase_average_policies")


######################################################### All Policy plots for basecase


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
