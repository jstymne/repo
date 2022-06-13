import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #with open(f'traces/O_1.npy', 'rb') as f:
    #    O_1 = np.load(f, allow_pickle=True)
    #with open(f'traces/O_2.npy', 'rb') as f:
    #    O_2 = np.load(f, allow_pickle=True)
    with open(f'O_3.npy', 'rb') as f:
        O_3 = np.load(f, allow_pickle=True)
    #with open(f'traces/Z_1.npy', 'rb') as f:
    #    Z_1 = np.load(f, allow_pickle=True)
    #with open(f'traces/Z_2.npy', 'rb') as f:
    #    Z_2 = np.load(f, allow_pickle=True)
    with open(f'Z_3.npy', 'rb') as f:
        Z_3 = np.load(f, allow_pickle=True)

    #with open(f'new_Z3.npy', 'rb') as f:
    #    Z_3 = np.load(f, allow_pickle=True)
    #print(O_1.shape)
    #print(O_2.shape)
    #print(O_3.shape)
    #print(Z_1.shape)
    #print(Z_2.shape)
    #print(Z_3.shape)
    #print(Z_1[:,:,0])

#print(Z_1[0,:,:] == Z_1[1,:,:] )
plt.figure()

#print(Z_1[:,:,0])

#obs = O_1

#Z_3 = Z_3[0,0:2,:]
obs_dist = Z_3[0,:]
obs_dist_intrusion = Z_3[1,:]
print(obs_dist_intrusion)
#np.save("Z_3.npy",Z_3,allow_pickle=True)
#print("sum" + str(sum(Z_1[0,0,:])))
plt.figure()
# set width of bar
#barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))
"""
# Set position of bar on X axis
br = np.arange(len(obs))
br1 = [x for x in br]
br2 = [x + barWidth for x in br]
"""
# Make the plot
#plt.bar(br1, obs_dist, width = barWidth,
#        edgecolor ='grey',  label = "Observation probability in state 0 = no intrusion")
#plt.bar(br2, obs_dist_intrusion, width = barWidth,
#        edgecolor ='grey', label = "Observation probability in state 1 = intrusion")

#print(obs_dist[:-1])
#print(obs_dist_intrusion[:-1])

plt.plot(obs_dist[:-1], label = "Observation probability in state 0 = no intrusion")
plt.plot(obs_dist_intrusion[:-1], label = "Observation probability in state 1 = intrusion")

# Adding Xticks

plt.xlabel("# of severe IDS alters", fontweight ='bold', fontsize = 16)
plt.ylabel("Observation probability", fontweight ='bold', fontsize = 16)
#plt.xticks([r + barWidth/2 for r in range(len(obs))],
#        obs)

ax = plt.gca()
plt.legend(fontsize = 16, bbox_to_anchor=(0.67, 1.15), bbox_transform=ax.transAxes)
#plt.legend()
plt.savefig('obs_dist_plot_traces')
