import torch
from qucumber.nn_states import ComplexWaveFunction
from qucumber.callbacks import MetricEvaluator
import qucumber.utils.unitaries as unitaries
import qucumber.utils.data as data
import qucumber as qc
import matplotlib.pyplot as plt
import numpy as np
import os


class Tomography:
    def __init__(self, folder_name, period=100, seed=1234, gpu=True):
        """
        Reconstruct quantum states by using the RBM implementation of Qucumber. The measurements used for the training
        have to be generated using the class State first.

        Args:
            folder_name: Name of the folder where the data generated with the class State() was saved.
            period: Period of the data output during the learning. Default value period = 100.
            seed: Seed of the RBM. Default value seed = 1235
            gpu: If True, the GPU is used. Default value gpu = False.
        """
        qc.set_random_seed(seed, cpu=True, gpu=gpu)
        self.folder_name = "Training_data/" + folder_name
        self.period = period
        self.gpu = gpu
        # All of these variables will be initialized later on:
        self.epochs = None
        self.batchsize = None
        self.learning_rate = None
        self.k = None
        self.optimizer = None
        self.optimizer_args = None
        self.scheduler = None
        self.scheduler_args = None
        self.nv = None
        self.nh = None
        self.space = None
        self.RBM_state = None
        self.callbacks = None
        self.callbacks_dict = None
        self.RBMpsi = None
        # Load the data
        self.train_samples, self.true_psi, self.train_bases, self.bases = self.load_data()
        # Set the parameters for the plot.
        self.plot_params = {"text.usetex": True, "font.family": "serif",
                            "legend.fontsize": 14, "figure.figsize": (10, 3),
                            "axes.labelsize": 16, "xtick.labelsize": 14,
                            "ytick.labelsize": 14, "lines.linewidth": 3,
                            "lines.markeredgewidth": 0.8, "lines.markersize": 5,
                            "lines.marker": "o", "patch.edgecolor": "black"}

    def load_data(self):
        """
        Loads the data from the folder specified in self.folder.

        Returns:
            self.train_samples: Training samples loaded from the self.folder.
            self.true_psi: The statevector of the state to be reconstructed.
            self.train_bases: Training bases loaded from the self.folder.
            self.bases: List of all training bases in self.train_bases.
        """
        bases_path = self.folder_name + "/bases.txt"
        train_bases_path = self.folder_name + "/train_bases.txt"
        train_path = self.folder_name + "/samples.txt"
        try:
            psi_path = self.folder_name + "/psi.txt"
            self.train_samples, self.true_psi, self.train_bases, self.bases = data.load_data(
                train_path, psi_path, train_bases_path, bases_path)
            print(self.true_psi)
        except OSError:
            psi_path = None
            self.true_psi = None
            self.train_samples, self.train_bases, self.bases = data.load_data(
                train_path, psi_path, train_bases_path, bases_path)
        return self.train_samples, self.true_psi, self.train_bases, self.bases

    def define_RBM(self, callbacks_dict):
        """
        Defines the RBM structure and the callbacks.

        Args:
            callbacks_dict: dictionary of callback names and the respective functions, i.e. {"Fidelity": ts.fidelity}
        """
        self.nv = self.train_samples.shape[-1]
        self.nh = self.nv - 1

        # initialize the ComplexWaveFunction object
        unitary_dict = unitaries.create_dict()
        self.RBM_state = ComplexWaveFunction(num_visible=self.nv, num_hidden=self.nh, unitary_dict=unitary_dict,
                                             gpu=self.gpu)
        self.space = self.RBM_state.generate_hilbert_space()

        # Define the callbacks
        self.callbacks_dict = callbacks_dict
        self.callbacks = [MetricEvaluator(self.period, callbacks_dict,
                                          target=self.true_psi,
                                          bases=self.bases,
                                          verbose=True,
                                          space=self.space)]

    def fit(self, epochs, learning_rate):
        """
        Fit the RBM.

        Args:
            epochs: Number of epochs.
            learning_rate: Learning rate of the RBM.
        """
        # ensure that all parameters for the fit are specified. Otherwise use default values.
        if self.batchsize == None:
            self.batchsize = self.train_samples.shape[0]
        if self.k == None:
            self.k = 100
        if self.optimizer == None:
            self.optimizer = torch.optim.Adagrad
            self.optimizer_args = {}
        if self.scheduler == None:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR
            self.scheduler_args = {"milestones": [1000, 1500], "gamma": 0.75}

        # ensure that the number of epochs is a multiple of self.epochs:
        epochs = int(epochs / self.period) * self.period

        # Fit the RBM with the given parameters.
        self.RBM_state.fit(
            self.train_samples,
            epochs=epochs,
            pos_batch_size=self.batchsize,
            neg_batch_size=self.batchsize,
            lr=learning_rate,
            k=self.k,
            input_bases=self.train_bases,
            callbacks=self.callbacks,
            time=True,
            optimizer=self.optimizer,
            optimizer_args=self.optimizer_args,
            scheduler=self.scheduler,
            scheduler_args=self.scheduler_args
        )

    def fit_routine(self, epochs, batchsize, learning_rate, contrast_div_steps, optimizer,
                    optimizer_args, scheduler, scheduler_args, treshold = 0.9, path_from_before=None):
        """
        A routine which works well also for states which are difficult to learn by using self.fit().
        After the specified number of epochs it is verified if the fidelity is above 70%. If not, onother
        1000 epochs are added up to 10 times.

        Args:
            treshold: continue the learning up to three times with 500 epochs each if the fidelity is smaller than
                      treshold
            epochs: Number of epochs.
            batchsize: Batchsize. Best results are obtained if batchsize = length of the training samples, since
                       phase and amplitude are trained seperately. If not the phase is not learned fast enough.
            learning_rate: Learning rate. Best results are obtained if learning_rate = 0.05.
            contrast_div_steps: Number of cantrastive divergence steps.
            optimizer: Optimizer from torch. Best results are obtained if optimizer = torch.optim.Adagrad. Other
                       possibilities can be found here: https://pytorch.org/docs/stable/optim.html
            optimizer_args: Dictionary of arguments which can be given to the optimizer as specified here:
                            https://pytorch.org/docs/stable/optim.html
            scheduler: Scheduler from torch (see https://pytorch.org/docs/stable/optim.html) if needed. Best
                       results are obtained for scheduler = torch.optim.lr_scheduler.MultiStepLR
            scheduler_args: Dictionary of arguments which can be given to the scheduler as specified here:
                            https://pytorch.org/docs/stable/optim.html. Best results are obtained for
                           {"milestones": [2000], "gamma": 0.75}.
            path_from_before: If None, no weights are loaded from another learning process. If it is a string specifying
                              the path to a file with the saved weights of a learning process, these weights are used at
                              the start of the fitting.
        """
        self.epochs = epochs
        self.batchsize = batchsize
        self.k = contrast_div_steps
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.scheduler = scheduler
        self.scheduler_args = scheduler_args
        self.learning_rate = learning_rate

        # Print that learning starts.
        try:
            print("Learn " + self.folder_name.split("/")[0] + " (" + self.folder_name.split("/")[2]
                  + "): " + str(self.train_samples.shape[0]) + " samples.")
        except IndexError:
            print("Learn " + self.folder_name + ": " + str(self.train_samples.shape[0]) + " samples.")
        if path_from_before != None:
            self.RBM_state.load("Training_data/" + path_from_before + "/saved_params.pt")

        self.fit(epochs, learning_rate)

        self.RBM_state.save(self.folder_name + "/saved_params.pt")
        for i in range(0, 3):
            try:
                continue_loop = (self.callbacks[0]["Fidelity"][-2] <= treshold)
            except (ValueError, AttributeError) as error:
                continue_loop = False
            if continue_loop:
                self.RBM_state.load(self.folder_name + "/saved_params.pt")
                self.fit(self.period * 10, learning_rate * 2)
                self.epochs += self.period * 10
                self.RBM_state.save(self.folder_name + "/saved_params.pt")
        self.get_RBM_state()

    def get_RBM_state(self):
        """
        Gets the learned RBM state representation.

        Returns:
            self.RBMpsi: The RBM representation of the reconstructed state.
        """
        Z = self.RBM_state.normalization(self.space)
        psi = self.RBM_state.psi(self.space) / Z.sqrt()
        psi = psi.tolist()
        new_psi = []
        sum_of_all = 0
        for i in range(len(psi[0])):
            p = psi[0][i] + psi[1][i] * 1j
            new_psi.append(p)
            sum_of_all += np.abs(p) ** 2
        self.RBMpsi = new_psi / np.sqrt(sum_of_all)
        return self.RBMpsi

    def get_RBM_state_amplitudes(self):
        """
        Gets the learned RBM state representation's amplitudes.

        Returns:
             A list of the RBM state representation's amplitude.
        """
        Z = self.RBM_state.normalization(self.space)
        amplitudes = self.RBM_state.amplitude(self.space) / Z.sqrt()
        return amplitudes.tolist()

    def get_RBM_state_phases(self):
        """
        Gets the learned RBM state representation's phases.

        Returns:
            A list of the RBM state representation's phase.
        """
        phases = self.RBM_state.phase(self.space)
        return phases.tolist()

    def get_amplitude_gradient(self):
        """
        Gets the gradient of the RBM which calculated the state's amplitude.

        Returns:
            The gradient of the amplitude RBM.
        """
        grad = self.RBM_state.am_grads(self.space)
        return grad

    def get_phase_gradient(self):
        """
        Gets the gradient of the RBM which calculated the state's phase.

        Returns:
            The gradient of the phase RBM.
        """
        grad = self.RBM_state.ph_grads(self.space)
        return grad

    def plot_callbacks(self):
        """
        Plot all callbacks specified in self.callbacks and save the figure.
        """
        plt.rcParams.update(self.plot_params)
        plt.style.use("seaborn-deep")

        epoch = np.arange(self.period, self.epochs + 1, self.period)

        for item in list(self.callbacks_dict.keys()):
            fig = plt.figure()
            if item == "Fidelity":
                cb = self.callbacks[0][item] ** (1 / np.log2(len(self.true_psi)))
                plt.plot(epoch, cb, "o", color="C0", markeredgecolor="black")
                plt.ylabel(r"rescaled fidelities $f^{\frac{1}{N}}$")
            else:
                cb = self.callbacks[0][item]
                plt.plot(epoch, cb, "o", color="C0", markeredgecolor="black")
                plt.ylabel(item)

            plt.xlabel(r"Learning Steps")
            plt.tight_layout()
            plt.show()
            fig.savefig(self.folder_name + "/" + item + ".png", dpi=1200)

    def save_details(self, other_folder=None):
        """
        Save - the RBM state as saved_params.pt
             - the fidelity in fidelity.npy and the Kullback Leibler divergence in KL.npy
             - The details of the learning in infos.txt.

        Args:
            other_folder: If None, all files are saved in self.folder. If it is a string specifying an existing or non
                          existing folder, the files are saved in this folder.
        """

        if other_folder == None:
            name = self.folder_name
        else:
            name = "Training_data/" + other_folder
            if not os.path.exists(name):
                os.makedirs(name)

        self.RBM_state.save(name + "/saved_params.pt")
        for item in list(self.callbacks_dict.keys()):
    	    np.save(name + "/" + item + ".npy", self.callbacks[0][item])

        file = open(name + '/infos.txt', "w")
        file.write("---RBM details ---\n")
        file.write("number of samples:                " + str(len(self.train_samples)) + "\n")
        file.write("target_state:                     " + str(self.true_psi) + "\n")

        file.write("number of nodes in visible layer: " + str(self.nv) + "\n")
        file.write("number of nodes in hidden layer:  " + str(self.nh) + "\n")
        file.write("---Training ---\n")
        file.write("number of training epochs:              " + str(self.epochs) + "\n")
        file.write("batch size:                             " + str(self.batchsize) + "\n")
        file.write("learning rate:                          " + str(self.learning_rate) + "\n")
        file.write("number of contrastive divergence steps: " + str(self.k) + "\n")
        file.write("---Results ---\n")
        file.write("RBM state:                        " + str(self.RBMpsi) + "\n")
        file.write("RBM state phase:                  " + str(self.RBM_state.phase(self.space)) + "\n")
        file.write("RBM state amplitude:              " + str(self.RBM_state.amplitude(self.space)) + "\n")
        file.write("RBM state normalization:          " + str(self.RBM_state.compute_normalization(self.space)) + "\n")
        file.write("RBM state probability in z-basis: " + str(
            self.RBM_state.probability(self.space, self.RBM_state.compute_normalization(self.space))) + "\n")
        file.write("weights:                          " + str(self.RBM_state.weights))
        file.write("visible bias:                     " + str(self.RBM_state.visible_bias))
        file.write("hidden bias:                      " + str(self.RBM_state.hidden_bias))
        file.close()
