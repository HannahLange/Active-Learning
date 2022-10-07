import activelearning 
import state
import numpy as np
import pandas as pd
import random

# State definitions
h = 1
lamb = -1
mu = 0
state_type = "HeisenbergModel"
qubit_no = 8

name = state_type
tomography_state = state.MPSstate({"qc_type": state_type, "qubit_no": qubit_no, "Parameters": {"h": h, "lamb": lamb, "mu": mu}}, -2)


#Specify the parameters for the passive learning part.
density_threshold = 0.005 # threshold value of the density difference to decide when to stop the learning for the MPS states
fidelity_threshold = 0.9 # threshold value of the fidelity to decide when to stop the learning for the IBM states
max_sxsx_szsz_difference = 0.2
marshall = 0.9
period = 100  # period for the callbacks
epochs = 4000 # number of epochs
lr = 0.07     # learning rate
k = 100       # contrastive divergence steps
seed_no =  4  # number of RBMs for the active learning

#Specify the parameters for the active learning part.
n_samples = 10 # number of measurements in the reference basis
query_samples = 10 # number of measurements for each query
config_no = 2**qubit_no # number of different configurations from which the query can choose



# START THE TOMOGRAPHY --------------------------------------------------------

# sample for the first time and choose the default basis

active_lrng = activelearning.Activelearning(tomography_state, name, lr, k, seed_no)

active_lrng.sample_first_time(n_samples, query_samples, with_rotation=True)

RBM_psis, RBM_states, RBM_amplitudes, RBM_phases, spaces = active_lrng.RBM_procedure(epochs, period, 0)
#RBM_states[0].save("saved_params.pt")

# ask max. 30 queries
previous_queries = []
n_samples_q = []
for j in range(1, 30):
    # stop the learning if the fidelity is above the threshold or density/ correlator difference below thereshold
    s=0
    for RBM_no in range(len(active_lrng.RBMseeds)):
        if tomography_state.state_type == "MPSstate":
            if tomography_state.qc_type == "LatticeGaugeModel":
                if np.abs(active_lrng.callbacks["correlator_difference"][RBM_no][-1]) <= density_threshold:
                    s += 1
            elif tomography_state.qc_type == "HeisenbergModel":
                sxsx = (active_lrng.callbacks["SxSx"][RBM_no][-1])
                sysy = (active_lrng.callbacks["SySy"][RBM_no][-1])
                szsz = (active_lrng.callbacks["SzSz"][RBM_no][-1])
                target_sxsx = np.sum(np.loadtxt("Training_data/tmp/target_sxsx.txt"))
                target_sysy = np.sum(np.loadtxt("Training_data/tmp/target_sysy.txt"))
                target_szsz = np.sum(np.loadtxt("Training_data/tmp/target_szsz.txt"))
                m_sign = np.abs(active_lrng.callbacks["Marshall"][RBM_no][-1]) 
                if sxsx < target_sxsx/3*2 and sysy < target_sysy/3*2 and szsz > target_szsz/3*2: # and m_sign >= marshall:
                    s += 1
            else:
                if active_lrng.callbacks["rescaled_fidelity"][RBM_no][-1] >= fidelity_threshold:
                    s+= 1
        else:
            if active_lrng.callbacks["rescaled_fidelity"][RBM_no][-1] >= fidelity_threshold:
                s += 1
    if s >= int(len(active_lrng.RBMseeds)/2) + 1:
        print("Threshold value is reached. Finish Learning.")
        active_lrng.query_type = "query_by_amplitude_and_phase"
        break
    query = active_lrng.ask_query_by_amplitude_and_phase(config_no, RBM_psis, RBM_states, RBM_amplitudes, RBM_phases, spaces)
    print(previous_queries)
    if len(previous_queries) != 0:
        x_config = ["X" for item in range(qubit_no)]
        x_config = "".join(x_config)
        y_config = ["Y" for item in range(qubit_no)]
        y_config = "".join(y_config)
        z_config = ["Z" for item in range(qubit_no)]
        z_config = "".join(z_config)
        if len(previous_queries) == 1:
            if previous_queries[0] == z_config:
                query = x_config
                active_lrng.n_samples_query = n_samples+3*query_samples
        elif len(previous_queries) != 1:
            if (previous_queries[-1] == z_config and previous_queries[-2] == z_config):
                query = x_config
                active_lrng.n_samples_query = n_samples+3*query_samples
            elif previous_queries[-1] == x_config and previous_queries[-2] == z_config:
                query = y_config
                active_lrng.n_samples_query = n_samples+3*query_samples
        else:
            active_lrng.n_samples_query = query_samples
    print("------------------- query: " + query + "-------------------")
    n_samples_query = active_lrng.sample_query(query, j)
    n_samples_q.append(n_samples_query)
    print(active_lrng.all_bases)
    RBM_psis, RBM_states, RBM_amplitudes, RBM_phases, spaces = active_lrng.RBM_procedure(epochs, period, j)
    previous_queries.append(query)

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


