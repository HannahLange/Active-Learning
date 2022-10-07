import activelearning 
import state
import numpy as np
import pandas as pd
import random
import math


random.seed(1013)
# State definitions
h = 0
lamb = -1
mu = 0
state_type = "GHZ"
qubit_no = 5

name = state_type
#tomography_state = state.MPSstate({"qc_type": state_type, "qubit_no": qubit_no, "Parameters": {"h": h, "lamb": lamb, "mu": mu}})

IBMtoken = "00ac9fd4c87a04cdfba4b3b015212069caf77663f7e5e83739029782ff3cf76c7c0dbe84132b0e55be620db6051e31663178c219624be95f1267298e1e48abc9"
tomography_state = state.IBMstate(IBMtoken, {"qc_type": state_type, "qubit_no": qubit_no}, "quantum")

#Specify the parameters for the passive learning part.
density_threshold = 0.2 # threshold value of the density difference to decide when to stop the learning for the MPS states
fidelity_threshold = 0.90 # threshold value of the fidelity to decide when to stop the learning for the IBM states
period = 100  # period for the callbacks
epochs = 1000 # number of epochs
lr = 0.07     # learning rate
k = 100       # contrastive divergence steps
seed_no =  4  # number of RBMs for the active learning

#Specify the parameters for the active learning part.
n_samples = 100 # number of measurements in the reference basis
query_samples = 1 # number of measurements for each query
config_no = 2**qubit_no # number of different configurations from which the query can choose



# START THE TOMOGRAPHY --------------------------------------------------------

# sample for the first time and choose the default basis

active_lrng = activelearning.Activelearning(tomography_state, name, lr, k, seed_no)

active_lrng.sample_first_time(n_samples, query_samples, with_rotation=True)

RBM_psis, RBM_states, RBM_amplitudes, RBM_phases, spaces = active_lrng.RBM_procedure(epochs, period, 0)
#RBM_states[0].save("saved_params.pt")

# ask max. 30 queries
n_samples_q = []
for j in range(1, 30):
    # stop the learning if the fidelity is above the threshold or density/ correlator difference below thereshold
    s=0
    for RBM_no in range(len(active_lrng.RBMseeds)):
        if tomography_state.state_type == "MPSstate":
            if np.abs(active_lrng.callbacks["correlator_difference"][RBM_no][-1]) <= density_threshold:
                s += 1
        else:
            if active_lrng.callbacks["rescaled_fidelity"][RBM_no][-1] >= fidelity_threshold:
                s += 1
    if s >= int(len(active_lrng.RBMseeds)/2) + 1:
        print("Threshold value is reached. Finish Learning.")
        active_lrng.query_type = "query_by_amplitude_and_phase"
        break
    query = active_lrng.ask_query_by_amplitude_and_phase(config_no, RBM_psis, RBM_states, RBM_amplitudes, RBM_phases, spaces)
    print("------------------- query: " + query + "-------------------")
    n_samples_query = active_lrng.sample_query(query, j)
    n_samples_q.append(n_samples_query)
    print(active_lrng.all_bases)
    RBM_psis, RBM_states, RBM_amplitudes, RBM_phases, spaces = active_lrng.RBM_procedure(epochs, period, j)


# plot the result
#active_lrng.plot_callbacks()
print("Save the Callbacks:")
for item in list(active_lrng.callbacks.keys()):
    print(item)
    df = pd.DataFrame(active_lrng.callbacks[item])
    df.to_csv(item + ".csv")

active_lrng.rotate_back()
with open('all_bases.txt', 'w') as f:
    for item in active_lrng.all_bases:
        f.write(item+"\n")

# baseline learning (same number of samples and measurement configurations without
# active learning.
print("Start the baseline reconstruction.")
bl_cb = {}
for i in range(seed_no):
    if i==0:
        active_lrng.sample_baseline(config_no)
    cb = active_lrng.get_baseline(epochs, period)
    bl_cb[i] = cb

for item in list(active_lrng.callbacks.keys()):
    baseline_cb = {}
    for i in range(seed_no):
        baseline_cb[i] = bl_cb[i][item]
    baseline_cb = pd.DataFrame(baseline_cb)
    baseline_cb.to_csv("baseline_"+item+".csv")


