import state
import tomography 
import qucumber.utils.training_statistics as ts
import torch
import pandas
IBMtoken = "b93de21e3b6d7f7973b4607b0c2ce1349618dee7edf3e037ab40c2a4d07bbd4db9b1ae59124c95dd8476adc47f4a790c2ec95045418bdd89990863beff6db6bb"



state_type = "1"
#configs = ["ZZZZZZZZ", "XYYXXXXX", "XXXXXXXX", "XYZXYZXX", "YYYYYYYY"] 
configs = ["ZZZZZ","XYYXX", "YYYYY","XXYXX", "XXXXX", "XYZXY"]
path = "Test"
qubit_no = 5

KL_dict = {}
fid_dict = {}
for i in range(4):
    KL_dict[i] = {}
    fid_dict[i] = {}
    for num_samples, sample_no in enumerate([100, 500, 1000, 2000, 5000]):
        example_state = state.IBMstate(IBMtoken, {"qc_type": state_type, "qubit_no": qubit_no})
        _, samples = example_state.generate_snapshots(sample_no, "equal", configs)
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
        KL_dict[i][sample_no] = tom.callbacks[0]["KL"]
        fid_dict[i][sample_no] = [f**(1/qubit_no) for f in tom.callbacks[0]["fidelity"]]
    #KL_dict[i] = pandas.DataFrame(KL_dict[i])
    #fid_dict[i] = pandas.DataFrame(fid_dict[i])

KL_dict = pandas.DataFrame(KL_dict)
fid_dict = pandas.DataFrame(fid_dict)
KL_dict.to_csv("KL.csv")
fid_dict.to_csv("fidelity.csv")
