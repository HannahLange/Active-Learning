import state
import tomography 
import qucumber.utils.training_statistics as ts
import torch
IBMtoken = "b93de21e3b6d7f7973b4607b0c2ce1349618dee7edf3e037ab40c2a4d07bbd4db9b1ae59124c95dd8476adc47f4a790c2ec95045418bdd89990863beff6db6bb"



state_type = "GHZ"
#configs = ["ZZZZZZZZ", "XYYXXXXX", "XXXXXXXX", "XYZXYZXX", "YYYYYYYY"] 
configs = ["ZZZZZZ","XYYXXX", "YYYYYY","XXYXXZ", "XXXXXX", "XYZXYZ"]
path = "Test"

KL_dict = {}
fid_dict = {}
KL_Qucumber_dict = {}
density_dict = {}
total_density_dict = {}
sigmax_diff_dict = {}
for num_samples, sample_no in enumerate([100, 500, 1000, 2000, 5000]):
    example_state = state.IBMstate(IBMtoken, {"qc_type": state_type, "qubit_no": 6})
    _, samples = example_state.generate_snapshots(600, "equal", configs)
    example_state.save_all(path)
    example_state.save_all("../")

    # Reconstruct the state

    callbacks = {"KL": ts.KL, "fidelity": ts.fidelity}
    tom = tomography.Tomography(path)
    batchsize = tom.train_samples.shape[0]
    tom.define_RBM(callbacks)


    # Fit the network
    epochs = 10000
    lr = 0.07
    k = 100
    tom.fit_routine(epochs, batchsize, lr, k, torch.optim.Adagrad, {}, torch.optim.lr_scheduler.MultiStepLR, {"milestones": [1000, 1500], "gamma": 0.75})
    tom.save_details()
    KL_dict[sample_no] = tom.callbacks[0]["KL"]
    fid_dict[sample_no] = tom.callbacks[0]["rescaled_fidelity"]

KL_dict = pandas.DataFrame(KL_dict)
fid_dict = pandas.DataFrame(fid_dict)
KL_dict.to_csv("KL.csv")
fid_dict.to_csv("fidelity.csv")
