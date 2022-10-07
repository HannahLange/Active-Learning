import qucumber
import qucumber.utils.training_statistics as ts
from qucumber.utils.unitaries import rotate_psi, rotate_psi_inner_prod
from qucumber.nn_states import ComplexWaveFunction, WaveFunctionBase
from qucumber.observables import ObservableBase, SigmaX, SigmaY, SigmaZ
from qucumber.observables.pauli import flip_spin
from qucumber.utils import cplx
from qucumber.observables.utils import _update_statistics, to_pm1
import qucumber.utils.data as data
from qucumber.observables import SigmaX, SigmaY, SigmaZ

import torch
from torch.distributions.utils import probs_to_logits

import itertools
from itertools import permutations, combinations_with_replacement
import random
import os
from qiskit import execute

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import shutil

import pyten as p
import state
import tomography 




class MySigmaXSigmaX(ObservableBase):
    """ Class for calculating the density observable. Will be used for the callbacks."""
    def __init__(self):
        self.name = "SigmaX"
        self.symbol = "X"

    def apply(self, nn_state, samples):
        """ Computes the value of the local-estimator of <S^x_i-1S^x_i>.
        Args:
            nn_state: The NeuralState that drew the samples.
            samples: A batch of sample states to calculate the observable on.
        Returns:
            real_sigmax: the real part of <S^x_i-1S^x_i>
            imag_sigmax: the imaginary part of <S^x_i-1S^x_i>
        """
        samples = samples.to(device=nn_state.device)

        # calculate the denominator of the local estimator (vector of shape: (2, num_samples,))
        denom = nn_state.importance_sampling_denominator(samples)

        # go through all sites
        real_sigmax = []
        imag_sigmax = []
        for i in range(samples.shape[-1]-1):
            # sigmax only gives contributions for opposite spins --> flip the spin at site i
            samples_ = flip_spin(i, samples.clone())
            samples_ = flip_spin(i+1, samples_.clone())
            # compute the numerator of the local estimator
            numer = nn_state.importance_sampling_numerator(samples_, samples)

            # get the total local estimator's real and imaginary parts
            mag = cplx.elementwise_division(numer, denom)
            real_mag = list(cplx.real(mag.cpu()))
            imag_mag = list(cplx.imag(mag.cpu()))
            # append them to a list
            real_sigmax.append(real_mag)
            imag_sigmax.append(imag_mag)
        # convert the lists to numpy arrays (shape =(no. of spins, no. of samples)) and transpose them)
        real_sigmax = np.array(real_sigmax).transpose()
        imag_sigmax = np.array(imag_sigmax).transpose()
        return real_sigmax, imag_sigmax

    def density_statistics(self, nn_state, num_samples, num_chains=0, burn_in=1000, steps=1, overwrite=False):
        """Estimates the expected value, variance, and the standard error of the
        densit
        Args:
            nn_state: The NeuralState to draw samples from.
            num_samples: The number of samples to draw. The actual number of
                            samples drawn may be slightly higher if
            num_chains: The number of Markov chains to run in parallel;
                           if 0 or greater than `num_samples`, will use a
                           number of chains equal to `num_samples`. This is not
                           recommended in the case where a `num_samples` is
                           large, as this may use up all the available memory.
                           Ignored if `initial_state` is provided.
            burn_in: The number of Gibbs Steps to perform before recording
                        any samples.
            steps: The number of Gibbs Steps to take between each sample.
            overwrite: Whether to overwrite the `initial_state` tensor, if
                          provided, with the updated state of the Markov chain.
        Returns:
            A dictionary with the mean, var and std of the density."""
        running_mean = 0.0
        running_variance = 0.0
        running_length = 0

        chains = None
        num_chains = (min(num_chains, num_samples) if num_chains != 0 else num_samples)

        num_time_steps = int(np.ceil(num_samples / num_chains))
        for i in range(num_time_steps):
            num_gibbs_steps = burn_in if i == 0 else steps

            chains = nn_state.sample(
                num_samples=num_chains,
                k=num_gibbs_steps,
                initial_state=chains,
                overwrite=True,
            )
            obs_samples_r, obs_samples_i = self.apply(nn_state, chains)
            mean = np.mean(obs_samples_r, axis=0)
            variance = np.var(obs_samples_r, axis=0)
            std_error = np.std(obs_samples_r, axis=0)
            sample_stats = {"mean": mean, "variance": variance, "std_error": std_error, "num_samples": len(obs_samples_r)}

            running_mean, running_variance, running_length = _update_statistics(
                running_mean,
                running_variance,
                running_length,
                sample_stats["mean"],
                sample_stats["variance"],
                num_chains,
            )
        std_error = np.sqrt(running_variance / running_length)

        return {
            "mean": running_mean,
            "variance": running_variance,
            "std_error": std_error,
            "num_samples": running_length,
        }


class MySigmaZSigmaZ(MySigmaXSigmaX):
    """ Class for calculating the density observable when the state is rotated to the X axis. Will be used for the callbacks."""
    def __init__(self):
        self.name = "SigmaX_in_X_Basis"
        self.symbol = "X_in_X_Basis"

    def apply(self, nn_state, samples):
        """ Computes the value of the local-estimator of <S^x_i-1S^x_i> within the rotated frame: <S^z_i-1S^z_i>.
        Args:
            nn_state: The NeuralState that drew the samples.
            samples: A batch of sample states to calculate the observable on.
        Returns:
            real_sigmax: the real part of <S^z_i-1S^z_i>
            imag_sigmax: the imaginary part of <S^z_i-1S^z_i>
        """
        samples = samples.to(device=nn_state.device)

        # go through all sites
        real_sigmax = []
        imag_sigmax = []
        for i in range(samples.shape[-1]-1):
            # calculate the value of s_z by converting the samples from 0 and 1s to (-1) and 1s.
            mag1 = to_pm1(samples[:, i])
            mag2 = to_pm1(samples[:, i+1])
            mag1 = cplx.make_complex(mag1, torch.zeros_like(mag1))
            mag2 = cplx.make_complex(mag2, torch.zeros_like(mag2))
            mag = cplx.elementwise_mult(mag1,mag2)
            real_mag = list(cplx.real(mag.cpu()))
            imag_mag = list(cplx.imag(mag.cpu()))
            # append them to a list
            real_sigmax.append(real_mag)
            imag_sigmax.append(imag_mag)
        # convert the lists to numpy arrays (shape =(no. of spins, no. of samples)) and transpose them)
        real_sigmax = np.array(real_sigmax).transpose()
        imag_sigmax = np.array(imag_sigmax).transpose()
        return real_sigmax, imag_sigmax


class MySigmaYSigmaY(MySigmaXSigmaX):
    """ Class for calculating the density observable. Will be used for the callbacks."""
    def __init__(self):
        self.name = "SigmaX"
        self.symbol = "X"

    def apply(self, nn_state, samples):
        """ Computes the value of the local-estimator of <S^x_i-1S^x_i>.
        Args:
            nn_state: The NeuralState that drew the samples.
            samples: A batch of sample states to calculate the observable on.
        Returns:
            real_sigmax: the real part of <S^y_i-1S^y_i>
            imag_sigmax: the imaginary part of <S^y_i-1S^y_i>
        """
        samples = samples.to(device=nn_state.device)

        # calculate the denominator of the local estimator (vector of shape: (2, num_samples,))
        denom = nn_state.importance_sampling_denominator(samples)

        # go through all sites
        real_sigmax = []
        imag_sigmax = []
        for i in range(samples.shape[-1]-1):
            coeff1 = to_pm1(samples[..., i]) # -i for spin down, i for spin_up)
            coeff1 = cplx.make_complex(torch.zeros_like(coeff1), coeff1)
            coeff2 = to_pm1(samples[..., i+1]) # -i for spin down, i for spin_up)
            coeff2 = cplx.make_complex(torch.zeros_like(coeff2), coeff2)
            coeff = cplx.elementwise_mult(coeff1, coeff2)
            # sigmay only gives contributions for opposite spins --> flip the spin at site i
            samples_ = flip_spin(i, samples.clone())
            samples_ = flip_spin(i+1, samples_.clone())
            # compute the numerator of the local estimator
            numer = nn_state.importance_sampling_numerator(samples_, samples)

            # get the total local estimator's real and imaginary parts
            mag = cplx.elementwise_mult(cplx.elementwise_division(numer, denom), coeff)
            real_mag = list(cplx.real(mag.cpu()))
            imag_mag = list(cplx.imag(mag.cpu()))
            # append them to a list
            real_sigmax.append(real_mag)
            imag_sigmax.append(imag_mag)
        # convert the lists to numpy arrays (shape =(no. of spins, no. of samples)) and transpose them)
        real_sigmax = np.array(real_sigmax).transpose()
        imag_sigmax = np.array(imag_sigmax).transpose()
        return real_sigmax, imag_sigmax



class MyCorr(MySigmaXSigmaX):
    """ Class for calculating the correlator observable. Will be used for the callbacks."""
    def __init__(self):
        self.name = "Corr"
        self.symbol = "C"

    def apply(self, nn_state, samples):
        """ Computes the value of the local-estimator of corr= <Sz_{i+1}...Sz_{i+d}*0.5*(1+Sx_i*Sx_{i+1})0.5*(1-Sx_{i+d}*Sx_{i+d+1})>.
        Args:
            nn_state: The NeuralState that drew the samples.
            samples: A batch of sample states to calculate the observable on.
        Returns:
            real_sigmax: the real part of corr = <Sz_{i+1}...Sz_{i+d}*0.5*(1+Sx_i*Sx_{i+1})0.5*(1-Sx_{i+d}*Sx_{i+d+1})>
            imag_sigmax: tensor with zeros.
        """
        samples = samples.to(device=nn_state.device)

        # calculate the denominator of the local estimator (vector of shape: (2, num_samples,))
        denom = nn_state.importance_sampling_denominator(samples)

        correlator = []
        # sigmax_i sigmax_i+1
        samples_2 = flip_spin(int(samples.shape[-1]/2)-1, samples.clone())
        samples_2 = flip_spin(int(samples.shape[-1]/2), samples_2.clone())
        sxi_sxip1 = nn_state.importance_sampling_numerator(samples_2, samples)
        for qubit_i in range(int(samples.shape[-1]/2),samples.shape[-1]-1):
            #sigmaz product
            corr1 = self.calculate_sigma_z_product(samples, qubit_i)
            #sigmax_j-1 sigmax_j
            samples_3 = flip_spin(qubit_i, samples.clone())
            samples_3 = flip_spin(qubit_i+1, samples_3.clone())
            sxipd_sxipdp1 = nn_state.importance_sampling_numerator(samples_3, samples)
            #sxsxsxsx
            samples_4 = flip_spin(qubit_i, samples_2.clone())
            samples_4 = flip_spin(qubit_i+1, samples_4.clone())
            sxsxsxsx = nn_state.importance_sampling_numerator(samples_4, samples)
            # get the contributions to the correlator
            corr2 = cplx.elementwise_division(cplx.elementwise_mult(self.calculate_sigma_z_product(samples_2,qubit_i), sxi_sxip1), denom)
            corr3 = cplx.elementwise_division(cplx.elementwise_mult(self.calculate_sigma_z_product(samples_3, qubit_i), sxipd_sxipdp1), denom)
            corr4 = cplx.elementwise_division(cplx.elementwise_mult(self.calculate_sigma_z_product(samples_4, qubit_i), sxsxsxsx), denom)
            corr1 = list(cplx.real(corr1.cpu()))
            corr2 = list(cplx.real(corr2.cpu()))
            corr3 = list(cplx.real(corr3.cpu()))
            corr4 = list(cplx.real(corr4.cpu()))
            # get the total local estimator's real part
            corr = [0.25*(corr1[i]+corr2[i]-corr3[i]-corr4[i]) for i, item in enumerate(corr2)]
            # append them to a list
            correlator.append(corr)
        # convert the lists to numpy arrays (shape =(no. of spins, no. of samples)) and transpose them)
        correlator = np.array(correlator).transpose()
        return correlator, cplx.make_complex(torch.zeros_like(sxsxsxsx), torch.zeros_like(sxsxsxsx))
    
    def calculate_sigma_z_product(self, samples, qubit_i):
        sigmaz_ip1 = to_pm1(samples[:, int(samples.shape[-1]/2)])
        sigmaz_ip1 = cplx.make_complex(sigmaz_ip1, torch.zeros_like(sigmaz_ip1))
        sigmaz_prod = sigmaz_ip1
        for j in range(int(samples.shape[-1]/2)+1, qubit_i+1):
            sigmaz_j = to_pm1(samples[:, j])
            sigmaz_j = cplx.make_complex(sigmaz_j, torch.zeros_like(sigmaz_j))
            sigmaz_prod = cplx.elementwise_mult(sigmaz_prod,sigmaz_j)
        return sigmaz_prod


class MyCorr_in_X_Basis(MySigmaXSigmaX):
    """ Class for calculating the correlator observable when the frame is rotated to the X basis. Will be used for the callbacks."""
    def __init__(self):
        self.name = "Corr_in_X"
        self.symbol = "C_in_X"

    def apply(self, nn_state, samples):
        """ Computes the value of the local-estimator of corr= <Sx_{i+1}...Sx_{i+d}*0.5*(1+Sz_i*Sz_{i+1})0.5*(1-Sz_{i+d}*Sz_{i+d+1})>.
        Args:
            nn_state: The NeuralState that drew the samples.
            samples: A batch of sample states to calculate the observable on.
        Returns:
            real_sigmax: the real part of corr= <Sx_{i+1}...Sx_{i+d}*0.5*(1+Sz_i*Sz_{i+1})0.5*(1-Sz_{i+d}*Sz_{i+d+1})>.
            imag_sigmax: tensor with zeros.
        """
        samples = samples.to(device=nn_state.device)

        # calculate the denominator of the local estimator (vector of shape: (2, num_samples,))
        denom = nn_state.importance_sampling_denominator(samples)

        correlator = []
        # sigmax only gives contributions for opposite spins --> flip the spin at site i
        # Sz_{i}*Sz_{i+1}
        sigmaz_i = to_pm1(samples[:, int(samples.shape[-1]/2)-1])
        sigmaz_i = cplx.make_complex(sigmaz_i, torch.zeros_like(sigmaz_i))
        sigmaz_ip1 = to_pm1(samples[:, int(samples.shape[-1]/2)])
        sigmaz_ip1 = cplx.make_complex(sigmaz_ip1, torch.zeros_like(sigmaz_ip1))
        sigmaz_i_ip1 = cplx.elementwise_mult(sigmaz_i, sigmaz_ip1)
        for qubit_i in range(int(samples.shape[-1]/2),samples.shape[-1]-1):
            #sigmax product
            corr1 = self.calculate_sigma_x_product(samples, qubit_i, nn_state, denom)
            # Sz_{i+d}*Sz_{i+d+1}
            sigmaz_j = to_pm1(samples[:, qubit_i])
            sigmaz_j = cplx.make_complex(sigmaz_j, torch.zeros_like(sigmaz_j))
            sigmaz_jp1 = to_pm1(samples[:, qubit_i+1])
            sigmaz_jp1 = cplx.make_complex(sigmaz_jp1, torch.zeros_like(sigmaz_jp1))
            sigmaz_j_jp1 = cplx.elementwise_mult(sigmaz_j, sigmaz_jp1)
            # Sz_{i}*Sz_{i+1} Sz_{i+d}*Sz_{i+d+1}
            szszszsz = cplx.elementwise_mult(sigmaz_i_ip1, sigmaz_j_jp1)
            # get the contributions to the correlator
            corr2 = cplx.elementwise_mult(self.calculate_sigma_x_product(samples, qubit_i, nn_state, denom), sigmaz_i_ip1)
            corr3 = cplx.elementwise_mult(self.calculate_sigma_x_product(samples, qubit_i, nn_state, denom), sigmaz_j_jp1)
            corr4 = cplx.elementwise_mult(self.calculate_sigma_x_product(samples, qubit_i, nn_state, denom), szszszsz)
            corr1 = list(cplx.real(corr1.cpu()))
            corr2 = list(cplx.real(corr2.cpu()))
            corr3 = list(cplx.real(corr3.cpu()))
            corr4 = list(cplx.real(corr4.cpu()))
            # get the total local estimator's real part
            corr = [0.25*(corr1[i]+corr2[i]-corr3[i]-corr4[i]) for i, item in enumerate(corr2)]
            # append them to a list
            correlator.append(corr)
        # convert the lists to numpy arrays (shape =(no. of spins, no. of samples)) and transpose them)
        correlator = np.array(correlator).transpose()
        return correlator, cplx.make_complex(torch.zeros_like(szszszsz), torch.zeros_like(szszszsz))
    
    def calculate_sigma_x_product(self, samples, qubit_i, nn_state, denom):
        samples_ = flip_spin(int(samples.shape[-1]/2), samples.clone())        
        for j in range(int(samples.shape[-1]/2)+1, qubit_i+1):
            samples_ = flip_spin(j, samples_.clone())
        num = nn_state.importance_sampling_numerator(samples_, samples)
        sigmax_prod = cplx.elementwise_division(num, denom)
        return sigmax_prod
    


class MyCorr_in_Y_Basis(MySigmaXSigmaX):
    """ Class for calculating the correlator observable. Will be used for the callbacks."""
    def __init__(self):
        self.name = "Corr_in_Y"
        self.symbol = "C_in_Y"

    def apply(self, nn_state, samples):
        """ Computes the value of the local-estimator of corr= <Sy_{i+1}...Sy_{i+d}*0.5*(1+Sx_i*Sx_{i+1})0.5*(1-Sx_{i+d}*Sx_{i+d+1})>.
        Args:
            nn_state: The NeuralState that drew the samples.
            samples: A batch of sample states to calculate the observable on.
        Returns:
            real_sigmax: the real part of corr = <Sy_{i+1}...Sy_{i+d}*0.5*(1+Sx_i*Sx_{i+1})0.5*(1-Sx_{i+d}*Sx_{i+d+1})>
            imag_sigmax: tensor with zeros.
        """
        samples = samples.to(device=nn_state.device)

        # calculate the denominator of the local estimator (vector of shape: (2, num_samples,))
        denom = nn_state.importance_sampling_denominator(samples)

        correlator = []
        
        for qubit_i in range(int(samples.shape[-1]/2),samples.shape[-1]-1):
            #sigmay
            mag, samples_sy = self.calculate_sigma_y_product(samples, qubit_i)
            corr1 = cplx.elementwise_division(cplx.elementwise_mult(nn_state.importance_sampling_numerator(samples_sy, samples), mag), denom)
            # sigmax only gives contributions for opposite spins --> flip the spin at site i
            samples_2 = flip_spin(int(samples.shape[-1]/2)-1, samples.clone())
            samples_2 = flip_spin(int(samples.shape[-1]/2), samples_2.clone())
            mag, samples_sy = self.calculate_sigma_y_product(samples_2, qubit_i)
            sxi_sxip1 = cplx.elementwise_mult(nn_state.importance_sampling_numerator(samples_sy, samples), mag)
            samples_3 = flip_spin(qubit_i, samples.clone())
            samples_3 = flip_spin(qubit_i+1, samples_3.clone())
            mag, samples_sy = self.calculate_sigma_y_product(samples_3, qubit_i)
            sxipd_sxipdp1 = cplx.elementwise_mult(nn_state.importance_sampling_numerator(samples_sy, samples), mag)
            samples_4 = flip_spin(qubit_i, samples_2.clone())
            samples_4 = flip_spin(qubit_i+1, samples_4.clone())
            mag, samples_sy = self.calculate_sigma_y_product(samples_4, qubit_i)
            sxsxsxsx = cplx.elementwise_mult(nn_state.importance_sampling_numerator(samples_sy, samples), mag)
            # get the contributions to the correlator
            corr2 = cplx.elementwise_division(sxi_sxip1, denom)
            corr3 = cplx.elementwise_division(sxipd_sxipdp1, denom)
            corr4 = cplx.elementwise_division(sxsxsxsx, denom)
            corr1 = list(cplx.real(corr1.cpu()))
            corr2 = list(cplx.real(corr2.cpu()))
            corr3 = list(cplx.real(corr3.cpu()))
            corr4 = list(cplx.real(corr4.cpu()))
            # get the total local estimator's real part
            corr = [0.25*(corr1[i]+corr2[i]-corr3[i]-corr4[i]) for i, item in enumerate(corr2)]
            # append them to a list
            correlator.append(corr)
        # convert the lists to numpy arrays (shape =(no. of spins, no. of samples)) and transpose them)
        correlator = np.array(correlator).transpose()
        return correlator, cplx.make_complex(torch.zeros_like(sxsxsxsx), torch.zeros_like(sxsxsxsx))
    
    def calculate_sigma_y_product(self, samples, qubit_i):
        coeff = to_pm1(samples[:, int(samples.shape[-1]/2)])
        mag1 = cplx.make_complex(torch.zeros_like(coeff), coeff)
        samples_ = flip_spin(int(samples.shape[-1]/2), samples.clone())
        for j in range(int(samples.shape[-1]/2)+1, qubit_i+1):
            coeff = to_pm1(samples_[:, j])
            mag2 = cplx.make_complex(torch.zeros_like(coeff), coeff)
            mag1 = cplx.elementwise_mult(mag1,mag2)
            # sigmay only gives contributions for opposite spins --> flip the spin at site i
            samples_ = flip_spin(j, samples_.clone())            
        return mag1, samples_



class Activelearning:
    def __init__(self, tomography_state, folder_name, learning_rate, contrastive_div_steps, num_of_RBMs):
        """
        Combine the generation of measurements using Qiskit in the class State and the RBM state tomography using
        Qucumber in the class Tomography with active Learning. This class provides different query strategies.

        Args:
            tomography_state: An instance of the class IBMState or MPS state, which is the object we aim to reconstruct.
            folder_name: Name of the folder where everything is saved later on.
            learning_rate: Learning rate of the RBM.
            contrastive_div_steps: Contrastive divergence steps for the Gibbs sampling of the RBM learning.
            num_of_RBMs: Number of RBMs with different, randomly generated, seeds. They are needed to evaluate
                         appropriate queries.
        """
        self.state = tomography_state
        self.folder_name = folder_name
        self.learning_rate = learning_rate
        self.k = contrastive_div_steps
        # All of these variables will be initialized later on:
        self.query_type = "Non"
        self.n_samples = None
        self.n_samples_query = None
        self.all_samples = None
        self.all_bases = None
        self.all_train_bases = None
        self.period = None
        self.epochs = {}
        self.baseline_path = None
        self.best_basis = None
        self.with_rotation = True

        # Generate the seeds for the RBMs.
        self.RBMseeds = [19801,213, 59763, 1568, 29, 105679][:num_of_RBMs]
        # Initialize the callbacks dictionary for Fidelity, Kullback Leibler divergence and Phase.
        if self.state.state_type == "MPSstate":
            if self.state.qc_type == "LatticeGaugeModel":
                self.callbacks = {"KL": {}, "total_density": {}, "density_difference": {}, "correlator_difference": {}}
            elif self.state.qc_type == "HeisenbergModel":
                self.callbacks = {"KL":{}, "Sx":{}, "Sy":{}, "Sz":{}, "SxSx":{}, "SySy": {}, "SzSz": {}, "Marshall": {}}
            else:
                self.callbacks = {"KL": {}, "KL(Qucumber)": {}, "rescaled_fidelity": {}, "total_density": {}, "density_difference": {}}
        else:
            self.callbacks = {"rescaled_fidelity": {}, "KL": {}}
        for RBM_no in range(len(self.RBMseeds)):
            for items in list(self.callbacks.keys()):
                self.callbacks[items][RBM_no] = np.array([])
            self.epochs[RBM_no] = 0

        # Set the parameters for the plot.
        self.plot_params = {"text.usetex": True, "font.family": "serif",
                            "legend.fontsize": 14, "figure.figsize": (10, 3),
                            "axes.labelsize": 16, "xtick.labelsize": 14,
                            "ytick.labelsize": 14, "lines.linewidth": 3,
                            "lines.markeredgewidth": 0.8, "lines.markersize": 5,
                            "lines.marker": "o", "patch.edgecolor": "black"}

    def sample_first_time(self, n_samples, n_samples_query, with_rotation=True):
        """
        Samples the first measurements in Z...Z and X...X measurement configurations.

        Args:
            n_samples: Total number of samples to be generated.
            n_samples_query: Number of samples for each query. The same number of samples will be generated in
                             the X...X basis. In the Z...Z basis n_samples - n_samples_query will be generated.
            with_rotation: Boolean. If True, the default basis is rotated from the Z basis to X, Y or by pi/5 around the
                           X axis (see self.rotate_basis())
        """
        self.with_rotation = with_rotation
        self.n_samples = n_samples
        self.n_samples_query = n_samples_query
        # get the observables which can be compared to the ones of the reconstructed state
        if self.state.state_type == "MPSstate":
            sxsx, sysy, szsz, density, total_density, sigmax, correlator = self.state.get_observables()
            print("Observables are generated.")
        if with_rotation:
            # Rotate the state to different bases and look where the measurement outcomes
            # are most significant (vary most).
            print("------ Start GLOBAL rotations to different measurement configurations.------")
            self._rotate_basis()
            name = self.folder_name + "/withRotation/" + str(self.n_samples_query) + "samples" + "/" + str(
                self.n_samples) + "samples"

        else:
            # Generate two strings specifying the configurations with only Zs and only Xs as first configurations to sample.
            zz_config = ""
            for i in range(self.state.qubit_no):
                zz_config += "Z"
            setting = [zz_config]
            # Generate snapshots in these configurations.
            self.state.generate_snapshots(self.n_samples, probability="equal", measurement_setting=setting)
            name = self.folder_name + "/" + str(self.n_samples_query) + "samples" + "/" + str(self.n_samples) + "samples"
        # save everything temporary
        self.state.save_all("tmp")
        self.state.save_all("../")
        # Save them in the folder specified by name.
        self.state.save_all(name)
        # Directly load the data again, now everything is in the correct format for the Qucumber RBM.
        bases_path = "Training_data/" + name + "/bases.txt"
        train_bases_path = "Training_data/" +name + "/train_bases.txt"
        train_path = "Training_data/" + name + "/samples.txt"
        try:
            psi_path = "Training_data/" + name + "/psi.txt"
            self.all_samples, self.true_psi, self.all_train_bases, self.all_bases = data.load_data(
                train_path, psi_path, train_bases_path, bases_path)
        except OSError:
            psi_path = None
            self.true_psi = None
            self.all_samples, self.all_train_bases, self.all_bases = data.load_data(
                train_path, psi_path, train_bases_path, bases_path)

    def _rotate_basis(self):
        """Rotates the basis before the actual learning starts. The basis with the hightest variance in the amplitudes
        (= most informative) is taken as the default basis (here the amplitudes of the state are reconstructed)."""
        variances = []
        outputs = []

        # measure in different configurations and look where the variance of the wave 
        # functions of different RBMs is lowest
        variances = []
        data = {}
        #save the original (not rotated) states
        if self.state.state_type == "IBMstate":
            unrotated_state = self.state.qc.copy()
        if self.state.state_type == "MPSstate":
            self.state.qc.save("MPSstate.mps")
            self.state.lattice.save("MPSlattice.mps")
        #rotate
        for basis in ["Z","X","Y"]:
            print("Measure in "+basis+" basis.")
            data[basis] = {}
            if os.path.exists("Training_data/tmp/"):
                shutil.rmtree("Training_data/tmp/")
            if os.path.exists("Training_data/"+self.folder_name + "/withRotation/" + str(self.n_samples_query) + "samples" + "/" + str(self.n_samples) + "samples"):
                shutil.rmtree("Training_data/"+self.folder_name + "/withRotation/" + str(self.n_samples_query) + "samples" + "/" + str(self.n_samples) + "samples")
            data[basis]["samples"], data[basis]["all_train_bases"], data[basis]["all_bases"] = self._measure_rotated_state(basis)
            #print(self.all_samples)
            self.all_samples = data[basis]["samples"]
            #print(self.all_samples)
            self.all_train_bases = data[basis]["all_train_bases"]
            self.all_bases = data[basis]["all_bases"]
            RBM_psis, RBM_states, RBM_amplitudes, RBM_phases, spaces = self.RBM_procedure(200, 100, 0)
            data[basis]["callbacks"] = self.callbacks
            v_psis = 0
            for q in range(len(RBM_psis[0])):
                psis = []
                p_sum = 0
                for rbm in range(len(RBM_phases)):
                    p_sum += np.abs(RBM_psis[rbm][q])/len(RBM_psis)
                    psis.append(RBM_psis[rbm][q])
                v_psis += np.var(psis)/p_sum
            # remove all callbacks and samples
            for quantity in self.callbacks.keys():
                self.callbacks[quantity]={}
                for RBM_no in range(len(self.RBMseeds)):
                    self.callbacks[quantity][RBM_no] = np.array([])
                    self.epochs[RBM_no] = 0
            print(v_psis)
            variances.append(v_psis)
            self.state.training_samples = None
            self.state.training_bases = None
            self.state.bases = None
        print(variances)
        # Determine the measurement basis where the variance is the lowest.
        best_basis = np.where(np.array(variances) == min(variances))[0][0]
        # rotate the state to the best basis
        if best_basis == 0:
            self.best_basis = "Z"
            self.rotated_target_state = self.state.target_state
        if best_basis == 1:
            self.best_basis = "X"
            if self.state.state_type == "IBMstate":
                # rotate the state to the x basis
                for i in range(self.state.qubit_no):
                    self.state.measure_x(i, self.state.qc)
                self.rotated_target_state = self.state._get_state()
            else:
                self.state.qc = self.rotate_MPS_state("X")
        if best_basis == 2:
            self.best_basis = "Y"
            # rotate the state to the y basis
            if self.state.state_type == "IBMstate":
                for i in range(self.state.qubit_no):
                    self.state.measure_y(i, self.state.qc)
                self.rotated_target_state = self.state._get_state()
            else:
                self.state.qc = self.rotate_MPS_state("Y")
        print("Using the " + self.best_basis + " basis for the amplitude measurements.")

        # Save the measurement results in the appropriate format.
        self.state.bases = np.array(data[self.best_basis]["all_bases"])
        self.state.training_bases = np.array(data[self.best_basis]["all_train_bases"])
        self.state.training_samples = np.array(data[self.best_basis]["samples"])
        self.callbacks = data[self.best_basis]["callbacks"]

    def rotate_MPS_state(self, basis):
        for i in range(self.state.qubit_no):
            if basis == "X":
                rotation = [1/np.sqrt(2)*4*self.state.lattice.get("sz",i)*self.state.lattice.get("sz",i)-1/np.sqrt(2)*2j*self.state.lattice.get("sy",i)]
            elif basis == "Y":
                rotation = [1/np.sqrt(2)*4*self.state.lattice.get("sz",i)*self.state.lattice.get("sz",i)-1/np.sqrt(2)*2j*self.state.lattice.get("sx",i)]
            rotation = p.mp.addLog(rotation)
            p.mp.apply_op_itrunc(self.state.qc, rotation, p.Truncation(1e-16))
        return self.state.qc


    def _measure_rotated_state(self,basis):
        """
        Measure the state in the "A...A" configuration with A="X","Y","Z" and saves the samples in the folder A_basis

        Args:
            basis: configuration to measure in (X, Y, Z)
        """
        zz_config = ""
        config = ""
        for i in range(self.state.qubit_no):
           zz_config += "Z"
           config += basis
        setting = [config]
        self.state.generate_snapshots(self.n_samples, probability="equal", measurement_setting=setting)
        self.state.training_bases = np.array([zz_config for i in self.state.training_bases])
        self.state.bases = np.array([zz_config])
        # save everything temporary
        name = self.folder_name + "/withRotation/" + str(self.n_samples_query) + "samples" + "/" + str(self.n_samples) + "samples"
        self.state.save_all(name)
        self.state.save_all("tmp")
        bases_path = "Training_data/" + name + "/bases.txt"
        train_bases_path = "Training_data/" +name + "/train_bases.txt"
        train_path = "Training_data/" + name + "/samples.txt"
        psi_path = None
        true_psi = None
        all_samples, all_train_bases, all_bases = data.load_data(train_path, psi_path, train_bases_path, bases_path)
        print(all_samples)
        return all_samples, all_train_bases, all_bases
        

    def ask_query_by_variance(self, config_no, RBM_psis):
        """
        Generates the next query by sampling from the RBM wave functions for different RBMs in different measurement
        configurations and chooses the configuration where the variance of the measurements for different RBM is largest.

        Args:
            config_no: Number of measurement configurations to measure in. One of them is chosen as a query.
            RBM_psis: A list of the RBM state representations to be sampled from.

        Returns:
            query_output: The query.

        """
        self.query_type = "query_by_variance"
        # Get all possible configurations for the number of qubits self.state.qubit_no.
        pool_of_configs = self._get_all_configurations(config_no)
        # Randomly choose config_no configurations.
        configs = random.sample(pool_of_configs, config_no)
        # Find the variances for different RBMs for each configuration.
        distribution = self._get_measurement_distribution(RBM_psis, configs)
        variances = self._get_variances(distribution)
        # Choose the configuration with maximal variance.
        maximum = max(list(variances.values()))
        query_output = []
        for config in list(variances.keys()):
            if variances[config] == maximum:
                query_output.append(config)
        return query_output

    def ask_query_by_amplitude_and_phase(self, config_no, RBM_psis, RBM_states, RBM_amplitudes, RBM_phases, spaces):
        """
        Extension to self.ask_query_by_variance(): Firstly, the variation in the amplitude and phases of the RBM state
         representations are evaluated. If var(amplitudes) > var(phases), the Z...Z basis is chosen as the query output.
         Else, the procedure is the same as in self.ask_query_by_variance(): The next query is generatedby sampling
          from the RBM wave functions for different RBMs in different measurement configurations and the configuration
          for which the variance of the measurements for different RBM is largest is chosen.

        Args:
            config_no: Number of measurement configurations to measure in. One of them is chosen as a query.
            RBM_psis: A list of the RBM state representations to be sampled from.
            RBM_amplitudes: A list of the RBM state representations' amplitudes.
            RBM_phases: A list of the RBM state representations' phases.

        Returns:
            query_output: The query.

        """
        self.query_type = "query_by_amplitude_and_phase"

        # Find the variance of amplitudes and phases between different RBMs
        phases = []
        amplitude_variation = 0
        RBM_amplitudes = np.array(RBM_amplitudes)
        RBM_phases = np.array(RBM_phases)
        for i, new_amplitudes in enumerate(RBM_amplitudes.transpose()):
            amplitude_variation += np.var(new_amplitudes) / max(new_amplitudes)
            phase = RBM_phases.transpose()[i, :]
            phase = [abs(abs(p) % (2 * pi)) for p in phase]
            phases.append(sum(phase) / max(phase))
        phase_variation = np.var(phases)

        # If var(amplitudes) > var(phase) choose the basis with only Zs as query output.
        if amplitude_variation >= phase_variation:
            query_output = ""
            for i in range(self.state.qubit_no):
                query_output += "Z"
        else:
            # If not, do the same as in self.ask_query_by_variance().
            pool_of_configs = self._get_all_configurations(config_no)
            if config_no >= len(self.all_bases):
                config_no = config_no - len(self.all_bases)
            configs = random.sample(pool_of_configs, config_no)
            configs = list(set(np.concatenate((np.array(configs), self.all_bases), axis=0)))
            distribution = self._get_measurement_distribution(RBM_psis, RBM_states, spaces, configs)
            variances = self._get_variances(distribution)
            maximum = max(list(variances.values()))
            query_output = []
            for config in list(variances.keys()):
                if variances[config] == maximum:
                    query_output.append(config)
            query_output = query_output[0]
        return query_output

    def _get_all_configurations(self, num_configs):
        """
        Gets all the combinations of X, Y and Z configurations of length self.state.qubit_no.

        Returns:
            config_pool: List of all Configurations.
        """
        # Use combinations_with_replacement to find all combinations of X, Y, Z of length self.state.qubit_no.
        if num_configs==2**self.state.qubit_no:
            cc = [set(permutations(c)) for c in list(combinations_with_replacement("XYZ", self.state.qubit_no))]
            configuration_combinations = []
            for c in cc:
                for item in c:
                    configuration_combinations.append(item)
        else:
            cc = []
            for i in range(num_configs-1):
                cc.append([random.sample(["X", "Y", "Z"],1)[0] for i in range(self.state.qubit_no)])
            cc.append(["Z" for i in range(self.state.qubit_no)])
            configuration_combinations = cc
        # Convert the configurations to strings, i. e. [X, Y] to "XY".
        config_pool = []
        for config in configuration_combinations:
            new_config = ""
            for c in config:
                new_config += c
            config_pool.append(new_config)
        return config_pool

    def _get_measurement_distribution(self, RBM_psis, RBM_states, spaces, configs):
        """
        Gets the measurement distribution for statevectors specified by PBM_psis in different measurement configurations
        specified in configs.

        Args:
            RBM_psis: List of statevectors.
            configs: List of configurations, i.e. ["Z...X", "Y...Z"].

        Returns:
            measurement_distribution: A dictionary with structure {"Z...X": {(0...0: {"all_values": {0: value related
            to the 1st statevector in RBM_psis, 1: value related to the 2nd statevector in RBM_psis, ...}, "variance":
            variance of values of the dictionary "all_values"}, (0...1): ... }, "Y...Z": {...}}
        """

        # Find all combinations of 0s and 1s of length self.state.qubit_no (= all statevector bases).
        sbc = [set(permutations(c)) for c in list(combinations_with_replacement([0, 1], self.state.qubit_no))]
        statevector_basis_combinations = []
        for s in sbc:
            for item in s:
                statevector_basis_combinations.append(item)
        # Make a template for a dictionary with the structure {(0...0: {"all_values": {0: value related to the 1st
        # statevector in RBM_psis, 1: value related to the 2nd statevector in RBM_psis, ...}, "variance": variance
        # of values of the dictionary "all_values"}, (0...1): ... }
        statevector_basis_dict = {}
        for combination in statevector_basis_combinations:
            statevector_basis_dict[str(combination)] = {"all_values": {}, "variance": 0}
        measurement_distribution = {}
        # Create the full dictionary and fill in the values for "all_values".
        for p, RBM_state in enumerate(RBM_states):
            dictionary = self._sample_distribution(RBM_state, spaces[p], configs, p)
            for configuration in dictionary.keys():
                if p == 0:
                    measurement_distribution[configuration] = statevector_basis_dict
                for statevector_basis in statevector_basis_dict.keys():
                    try:
                        measurement_distribution[configuration][statevector_basis]["all_values"][p] \
                            = dictionary[configuration][statevector_basis]
                    except KeyError:
                        measurement_distribution[configuration][statevector_basis]["all_values"][p] = 0
        # Fill in the values for "variance".
        for configuration in measurement_distribution.keys():
            l = []
            for statevector_basis in measurement_distribution[configuration].keys():
                values = list(measurement_distribution[configuration][statevector_basis]["all_values"].values())
                variance = np.var(values)
                measurement_distribution[configuration][statevector_basis]["variance"] = variance
        return measurement_distribution

    def _sample_distribution(self, RBM_state, space, configs, RBM_no):
        """
        Samples the distribution used in self._get_measurement_distribution().

        Args:
            psi: State vector from which is sampled.
            configs: List of configurations in which psi is measured.
            RBM_no: Number of the RBM, specified in self._get_measurement_distribution() by p.

        Returns:
            measurement_distribution: A dictionary with structure {Configuration 1: {0...0: percentage of measurement
            outcomes 0...0, 0...1: percentage of measurement outcomes 0...1, ...}, Configuration 2: {...}, ...}
        """
        
        # Create the state defined by psi.
        measurement_distribution = {}
        for config in configs:
            rotated_psi = rotate_psi(RBM_state, config, space)
            Z = RBM_state.normalization(space)
            samples = (cplx.absolute_value(rotated_psi)**2)/ Z
            samples_dict = {}
            for index, state in enumerate(space):
                state_string = "("
                for s in state:
                    state_string += str(int(s.item())) + ", "
                state_string = state_string[:-2] + ")"
                samples_dict[state_string] = samples[index].item()
            #print("samples:")
            #print(samples_dict)
            measurement_distribution[config] = samples_dict
        return measurement_distribution

    def _get_variances(self, measurement_distribution):
        """
        Gets the variances from the dictionary generated in self._get_measurement_distribution().
        Args:
            measurement_distribution: The measurement distribution dictionary generated in
            self._get_measurement_distribution()

        Returns:
            variances: A list of the variances for all statevector bases 0...0, 0...1, etc.
        """
        variances = {}
        for configuration in measurement_distribution.keys():
            var = []
            for statevector_basis in measurement_distribution[configuration].keys():
                variance = measurement_distribution[configuration][statevector_basis]["variance"]
                var.append(variance)
            sum_of_variances = sum(var)
            variances[configuration] =sum_of_variances
        return variances

    def sample_query(self, config_from_query, query_no):
        """
        Measures the state in the configuration determined by self.ask_query_by_variance() or
        self.ask_query_by_amplitude_and_phase().

        Args:
            config_from_query: The query output by self.ask_query_by_variance() or self.ask_query_by_amplitude_and_phase()
            query_no: Number of the query (needed to save the measurement results in the appropriate folder).

        Returns:
            n_samples_query: Number of samples generated for the query (If the query is "ZZ...Z", the number of generated
            samples is three times larger than the value specified in self.n_samples_query.
        """

        # Generate a string specifying the configuration with only Zs.
        zz_config = ""
        for i in range(self.state.qubit_no):
            zz_config += "Z"
        # If the configuration is the same as the one with only Zs, increase the number of samples by a factor 3.
        if config_from_query == zz_config:
            n_samples_query = self.n_samples_query * 3
        else:
            n_samples_query = self.n_samples_query
        # Generate the samples and save them.
        self.state.generate_snapshots(n_samples_query, probability="equal", measurement_setting=[config_from_query])
        name = self.folder_name + "/" + str(self.n_samples_query) + "samples/" + self.query_type \
               + "/query" + str(query_no)
        self.state.save_all(name)
        # Directly re-load them. Now everything is in the correct format for the Qucumber RBM.
        bases_path = "Training_data/" + name + "/bases.txt"
        train_bases_path = "Training_data/" +name + "/train_bases.txt"
        train_path = "Training_data/" + name + "/samples.txt"
        try:
            psi_path = "Training_data/" + name + "/psi.txt"
            train_samples, _, train_bases, bases = data.load_data(
                train_path, psi_path, train_bases_path, bases_path)
        except OSError:
            psi_path = None
            self.true_psi = None
            train_samples, train_bases, bases = data.load_data(
                train_path, psi_path, train_bases_path, bases_path)
        # If only one sample is generated per query, the format has to be changed a bit.
        try:
            if train_bases[0] == "X" or train_bases[0] == "Y" or train_bases[0] == "Z":
                train_samples = [train_samples.tolist()]
                train_samples = torch.tensor(train_samples, dtype=torch.float64)
                train_bases = np.array([train_bases])
        except ValueError:
            pass
        # Add the new samples, training bases and bases to self.all_samples, self.all_train_bases and self.all_bases.
        self.all_samples = torch.cat([self.all_samples, train_samples], dim=0)
        self.all_train_bases = np.concatenate((self.all_train_bases, train_bases), axis=0)
        self.all_bases = np.concatenate((self.all_bases, bases), axis=0)
        # Shuffle the data and add the number of query samples to the total number of samples.
        self.shuffle_data()
        self.n_samples += n_samples_query
        return n_samples_query

    def shuffle_data(self):
        """
        Shuffles the training samples and the respective bases in self.all_samples and self.all_train_bases.
        """
        temp = list(zip(self.all_train_bases, self.all_samples))
        random.shuffle(temp)
        shuffled_train_bases, shuffled_samples = zip(*temp)
        self.all_train_bases = np.array([list(shuffled_train_bases[0])])
        self.all_samples = torch.tensor([shuffled_samples[0].tolist()], dtype=torch.float64)
        for i in range(1, len(shuffled_train_bases)):
            self.all_samples = torch.cat(
                [self.all_samples, torch.tensor([shuffled_samples[i].tolist()], dtype=torch.float64)], dim=0)
            self.all_train_bases = np.concatenate((self.all_train_bases,
                                                   np.array([list(shuffled_train_bases[i])])), axis=0)

    def RBM_procedure(self, epochs, period, query_no, path_from_before=None):
        """
        The RBM learning procedure performed after each query measurement (and after the first measurement).

        Args:
            epochs: number of epochs.
            period: Period of the data output during the learning.
            query_no: Number of the query (needed to save everything in the appropriate folder).
            path_from_before: If None, no weights are loaded from another learning process. If it is a string specifying
                              the path to a file with the saved weights of a learning process, these weights are used at
                              the start of the fitting.
        Returns:
            RBM_psis: The RBM representation of the state which we want to learn.
            RBM_amplitudes: The RBM representation's amplitude of the state which we want to learn.
            RBM_phases: The RBM representation's phase of the state which we want to learn.
        """
        # Define all functions needed for the callbacks-----------------------------------------------------------------
        def get_corr(RBM_state, **kwargs):
            if self.best_basis == "X":
                statistics = MyCorr_in_X_Basis().density_statistics(RBM_state, 10000)
            elif self.best_basis == "Y":
                statistics = MyCorr_in_Y_Basis().density_statistics(RBM_state, 10000)
            else:
                statistics = MyCorr().density_statistics(RBM_state, 10000)
            corr = statistics["mean"]
            print(corr)
            target_corr = np.loadtxt("Training_data/tmp/correlator.csv")
            print(target_corr)
            corr_string = ""
            target_corr_string = ""
            for i, c in enumerate(corr):
                corr_string += str(c) + ","
                target_corr_string += str(target_corr[i]) +","
            if os.path.exists("correlator_full_data.csv"):
                with open("correlator_full_data.csv", "a") as file_object:
                    file_object.write(corr_string + "\n")
                with open("target_correlator_full_data.csv", "a") as file_object:
                    file_object.write(target_corr_string + "\n")
            else:
                with open("correlator_full_data.csv", "w") as file_object:
                    file_object.write(corr_string + "\n")
                with open("target_correlator_full_data.csv", "w") as file_object:
                    file_object.write(target_corr_string + "\n")
            if not np.isclose(np.linalg.norm(target_corr),0,rtol=1e-10):
                return np.linalg.norm(corr-target_corr)/np.linalg.norm(target_corr)
            else:
                return np.linalg.norm(corr-target_corr)

        def get_corr2(RBM_state, **kwargs):
            if self.best_basis == "X":
                statistics = MyCorr_in_X_Basis().density_statistics(RBM_state, 1000)
            elif self.best_basis == "Y":
                statistics = MyCorr_in_Y_Basis().density_statistics(RBM_state, 1000)
            else:
                statistics = MyCorr2().density_statistics(RBM_state, 1000)
            corr = statistics["mean"]
            print("corr2")
            print(corr)
            target_corr = np.loadtxt("Training_data/tmp/correlator.csv")
            print(target_corr)
            corr_string = ""
            target_corr_string = ""
            for i, c in enumerate(corr):
                corr_string += str(c) + ","
                target_corr_string += str(target_corr[i]) +","
            if os.path.exists("correlator_full_data.csv"):
                with open("correlator_full_data.csv", "a") as file_object:
                    file_object.write(corr_string + "\n")
                with open("target_correlator_full_data.csv", "a") as file_object:
                    file_object.write(target_corr_string + "\n")
            else:
                with open("correlator_full_data.csv", "w") as file_object:
                    file_object.write(corr_string + "\n")
                with open("target_correlator_full_data.csv", "w") as file_object:
                    file_object.write(target_corr_string + "\n")
            return np.linalg.norm(corr-target_corr)/np.linalg.norm(target_corr)


        def get_density(RBM_state):
            if self.best_basis == "X":
                statistics = MySigmaZSigmaZ().density_statistics(RBM_state, 1000)
            if self.best_basis == "Y":
                statistics = MySigmaXSigmaX().density_statistics(RBM_state, 1000)
            else:
                statistics = MySigmaXSigmaX().density_statistics(RBM_state, 1000)
            r_sxsx = statistics["mean"]
            density = []
            for s in r_sxsx:
                d = 0.5*(1-s)
                density.append(d)
            return np.array(density)

        def get_density_distance(RBM_state, **kwargs):
            density = get_density(RBM_state)
            target_density = np.loadtxt("Training_data/tmp/target_density.csv")
            density_string = ""
            target_density_string = ""
            for i, d in enumerate(density):
                density_string += str(d) + ","
                target_density_string += str(target_density[i]) +","
            if os.path.exists("density_full_data.csv"):
                with open("density_full_data.csv", "a") as file_object:
                    file_object.write(density_string + "\n")
                with open("target_density_full_data.csv", "a") as file_object:
                    file_object.write(target_density_string + "\n")
            else:
                with open("density_full_data.csv", "w") as file_object:
                    file_object.write(density_string + "\n")
                with open("target_density_full_data.csv", "w") as file_object:
                    file_object.write(target_density_string + "\n")
            if not  np.isclose(np.linalg.norm(target_density), 0, rtol=1e-10):
                return np.linalg.norm(density-target_density)/np.linalg.norm(target_density)
            else:
                return np.linalg.norm(density-target_density)

        def get_total_density(RBM_state, **kwargs):
            if self.state.qc_type != "HeisenbergModel":
                density = get_density(RBM_state)
                total_density = 0
                for d in density:
                    total_density += d
            elif self.state.qc_type == "HeisenbergModel":
                new_samples = RBM_state.sample(k=100, num_samples=10000)
                if self.best_basis == "X":
                    total_density = SigmaX(absolute=True).statistics_from_samples(RBM_state, new_samples)["mean"]
                elif self.best_basis == "Y":
                    total_density = SigmaY(absolute=True).statistics_from_samples(RBM_state, new_samples)["mean"]
                else:
                    total_density = SigmaZ(absolute=True).statistics_from_samples(RBM_state, new_samples)["mean"]
            return total_density*self.state.qubit_no

        def get_sx(RBM_state, **kwargs):
            if self.best_basis == "X":
                sx = SigmaZ(absolute=False)
            elif self.best_basis == "Y":
                sx = SigmaY(absolute=False)
            else:
                sx = SigmaX(absolute=False)
            sx = sx.statistics(RBM_state, num_samples=10000, burn_in=100)["mean"]
            return sx

        def get_sy(RBM_state, **kwargs):
            if self.best_basis == "X":
                sy = SigmaX(absolute=False)
            elif self.best_basis == "Y":
                sy = SigmaZ(absolute=False)
            else:
                sy = SigmaY(absolute=False)
            sy = sy.statistics(RBM_state, num_samples=10000, burn_in=100)["mean"]
            return sy

        def get_sz(RBM_state, **kwargs):
            if self.best_basis == "X":
                sz = SigmaX(absolute=False)
            elif self.best_basis == "Y":
                sz = SigmaY(absolute=False)
            else:
                sz = SigmaZ(absolute=False)
            sz = sz.statistics(RBM_state, num_samples=10000, burn_in=100)["mean"]
            return sz

        def get_sxsx(RBM_state, **kwargs):
            if self.best_basis == "X":
                statistics = MySigmaZSigmaZ().density_statistics(RBM_state, 1000)
            elif self.best_basis == "Y":
                statistics = MySigmaXSigmaX().density_statistics(RBM_state, 1000)
            else:
                statistics = MySigmaXSigmaX().density_statistics(RBM_state, 1000)
            r_sxsx = statistics["mean"]
            with open("sxsx_full_data.csv", "a") as file_object:
                file_object.write(", ".join([str(item) for item in r_sxsx]) + "\n")
            sxsx = 1/4*np.sum(r_sxsx)
            return sxsx
        
        def get_sysy(RBM_state, **kwargs):
            if self.best_basis == "X":
                statistics = MySigmaYSigmaY().density_statistics(RBM_state, 1000)
            elif self.best_basis == "Y":
                statistics = MySigmaZSigmaZ().density_statistics(RBM_state, 1000)
            else:
                statistics = MySigmaYSigmaY().density_statistics(RBM_state, 1000)
            r_sysy = statistics["mean"]
            with open("sysy_full_data.csv", "a") as file_object:
                file_object.write(", ".join([str(item) for item in r_sysy]) + "\n")
            sysy = 1/4*np.sum(r_sysy)
            return sysy
    
        def get_szsz(RBM_state, **kwargs):
            if self.best_basis == "X":
                statistics = MySigmaXSigmaX().density_statistics(RBM_state, 1000)
            elif self.best_basis == "Y":
                statistics = MySigmaYSigmaY().density_statistics(RBM_state, 1000)
            else:
                statistics = MySigmaZSigmaZ().density_statistics(RBM_state, 1000)
            r_szsz = statistics["mean"]
            with open("szsz_full_data.csv", "a") as file_object:
                file_object.write(", ".join([str(item) for item in r_szsz]) + "\n")
            szsz = 1/4*np.sum(r_szsz)
            return szsz
        
        def get_MarshallSign(RBM_state, **kwargs):
            reference_state = torch.tensor([(-1)**i for i in range(self.state.qubit_no)])
            basis_states = RBM_state.generate_hilbert_space()
            correct_predictions = 0
            for state in basis_states:
                sign = np.exp(1j*(RBM_state.phase(state)-RBM_state.phase(reference_state)).item()/2)
                if sign >=0: sign = 1
                else: sign = -1
                n = 0
                for q, qubit in enumerate(state):
                    if q != 0 and q % 2 != 0:
                        if qubit == 1:
                            n += 1
                marshall_sign = (-1)**n
                if marshall_sign == sign:
                    correct_predictions += 1
            return correct_predictions/len(basis_states)
            

        def get_fidelity(nn_state, target, space=None, **kwargs):
            fid = ts.fidelity(nn_state, target)
            samples = np.loadtxt("Training_data/tmp/samples.txt")
            qubit_no = len(samples[0])
            fid = fid**(1/qubit_no)
            return fid

        def generateAllBinaryStrings(grouped_samples):
            n = len(list(grouped_samples.keys())[0])
            for i in range(n+1):
                for j, bits in enumerate(itertools.combinations(range(n), i)):
                    s = ['0'] * n
                    for bit in bits:
                        s[bit] = '1'
                    if i == 0 and j == 0:
                        all_binary_strings = [''.join(s)]
                    else:
                        all_binary_strings.append(''.join(s))

            dictionary = {}
            for bit in all_binary_strings:
                dictionary[int(bit)] = bit

            ordered_binary_strings = []
            keys = list(dictionary.keys())
            keys.sort()
            for bit in keys:
                ordered_binary_strings.append(dictionary[bit])
            return ordered_binary_strings

        def get_target_probs(grouped_samples, basis, total_samples_per_basis):
            binary_strings = generateAllBinaryStrings(grouped_samples[basis])
            probs = []
            for sample in binary_strings:
                if sample in list(grouped_samples[basis].keys()):
                    probs.append(grouped_samples[basis][sample]/total_samples_per_basis)
                else:
                    probs.append(0)
            probs = torch.tensor(probs)
            return probs


        def _single_basis_KL(target_probs, nn_probs):
            return torch.sum(target_probs * probs_to_logits(target_probs)) - torch.sum(
                target_probs * probs_to_logits(nn_probs)
            )

        def KL(nn_state, target, space=None, **kwargs):
            train_samples = np.loadtxt("Training_data/tmp/samples.txt")
            train_bases   = list(np.loadtxt("Training_data/tmp/train_bases.txt", dtype='str'))
            grouped_samples = {}  # collect the same samples in each basis
            ordered_samples = {}  # get samples in each measurement basis
            bases = []
            samples = []
            for i, basis in enumerate(train_bases):
                string_sample = ""
                string_basis = ""
                for b in basis:
                    string_basis += b
                for s in train_samples[i]:
                    string_sample += str(int(s))
                bases.append(string_basis)
                samples.append(string_sample)
                if string_basis not in list(ordered_samples.keys()):
                    ordered_samples[string_basis] = []
                    grouped_samples[string_basis] = {}
                if string_sample not in list(grouped_samples[string_basis].keys()):
                    grouped_samples[string_basis][string_sample] = 0.0
                ordered_samples[string_basis].append(samples[i])
                grouped_samples[string_basis][string_sample] += 1.0

            space = space if space is not None else nn_state.generate_hilbert_space()
            Z = nn_state.normalization(space)
            KL = 0.0
            if isinstance(nn_state, WaveFunctionBase):
                for basis in set(bases):
                    psi_r = rotate_psi(nn_state, basis, space)
                    nn_probs_r = (cplx.absolute_value(psi_r) ** 2)  / Z
                    nn_probs_r = nn_probs_r.cpu()
                    target_probs_r = get_target_probs(grouped_samples, basis, len(ordered_samples[basis])).cpu()
                    KL += _single_basis_KL(target_probs_r, nn_probs_r)
                KL /= float(len(set(bases)))
            return KL.item()
        # -------------------------------------------------------------------------------------------------

        # Specify where the data is loaded from. Different for the first sampling than for the queries.
        if query_no == 0:
            if self.with_rotation:
                save_in = self.folder_name + "/withRotation/" + str(self.n_samples_query) + "samples" + "/" \
                          + str(self.n_samples) + "samples"
            else:
                save_in = self.folder_name + "/" + str(self.n_samples_query) + "samples" + "/" \
                          + str(self.n_samples) + "samples"
        else:
            save_in = self.folder_name + "/" + str(self.n_samples_query) + "samples/" + self.query_type \
                      + "/query" + str(query_no)

        # Loop over all RBMs with different seeds.
        RBM_psis = []
        RBM_amplitudes = []
        RBM_phases = []
        RBM_states = []
        spaces = []
        for i, seed in enumerate(self.RBMseeds):
            # Create a tomography object and initialize the training data by using the combination of new and old
            # training data.
            tom = tomography.Tomography(save_in, period, seed)
            tom.train_samples = self.all_samples
            tom.train_bases = self.all_train_bases
            tom.bases = self.all_bases

            # Choose the batchsize equal to the length of the dataset (since phase and amplitude of the
            # RBM state are trained seperately). Otherwise the phase is not learned fast enough.
            if self.state.state_type == "MPSstate":
                if self.state.qc_type == "LatticeGaugeModel":
                    callbacks = {"KL": KL, "total_density": get_total_density, "density_difference": get_density_distance, "correlator_difference": get_corr}
                elif self.state.qc_type == "HeisenbergModel":
                    callbacks = {"KL": KL, "Sx": get_sx, "Sy": get_sy, "Sz": get_sz, "SxSx":get_sxsx, "SySy":get_sysy, "SzSz": get_szsz, "Marshall": get_MarshallSign}
                else:
                    callbacks = {"KL": KL, "KL(Qucumber)": ts.KL, "rescaled_fidelity": get_fidelity, "total_density": get_total_density,"density_difference": get_density_distance}
            else:
                callbacks = {"rescaled_fidelity": get_fidelity, "KL": ts.KL}
            batchsize = tom.train_samples.shape[0]
            tom.define_RBM(callbacks, self.state.state_type)

            # If there is a path_from_before which is not None, complete the string such that everything works.
            if path_from_before != None:
                path_from_before_complete = path_from_before + "/" + str(i)
            else:
                path_from_before_complete = None

            # Fit the network.
            tom.fit_routine(epochs, batchsize, self.learning_rate, self.k, torch.optim.Adagrad, {},
                            torch.optim.lr_scheduler.MultiStepLR, {"milestones": [1000, 1500], "gamma": 0.75},
                            path_from_before_complete)

            #print(tom.RBMpsi)
            # Append the RBM statevextor, its amplitudes and phases.
            RBM_psis.append(tom.get_RBM_state())
            RBM_states.append(tom.RBM_state)
            RBM_amplitudes.append(tom.get_RBM_state_amplitudes())
            RBM_phases.append(tom.get_RBM_state_phases())
            spaces.append(tom.space)
            # remove the files from before if not needed
            if path_from_before == None:
                if i >= 1:
                    for file in os.listdir("Training_data/" + save_in + "/" + str(i - 1)):
                        os.remove("Training_data/" + save_in + "/" + str(i - 1) + "/" + file)
                    os.rmdir("Training_data/" + save_in + "/" + str(i - 1))
            # save the details of the RBM
            tom.save_details(save_in + "/" + str(i))
            # Add callbacks to the self.callbacks dictionary.
            for quantity in list(self.callbacks.keys()):
                self.callbacks[quantity][i] = np.hstack((self.callbacks[quantity][i], tom.callbacks[0][quantity]))
            self.epochs[i] += tom.epochs
        self.period = period
        return RBM_psis, RBM_states, RBM_amplitudes, RBM_phases, spaces

    def rotate_back(self):
        if self.with_rotation:
            if self.state.state_type == "IBMstate":
                if self.best_basis == "X":
                    for i in range(self.state.qubit_no):
                        self.state.measure_x(i, self.state.qc)
                if self.best_basis == "Y":
                    for i in range(self.state.qubit_no):
                        self.state.qc.sdg(i)
                        self.state.qc.h(i)                        
                self.rotated_target_state = self.state._get_state()

            else:
                p.cpp_pyten.mp.MPS.load(self.state.qc, "MPSstate.mps")
                p.cpp_pyten.mp.Lattice.load(self.state.lattice, "MPSlattice.mps")
                if not os.path.exists("baseline_not_rotated"):
                    os.makedirs("baseline_not_rotated")
                sxsx, sysy, szsz, density, total_density, sigmax, correlator = self.state.get_observables()
                self.state.save_all("../baseline_not_rotated/")

    def get_baseline(self, epochs, period):
        """
        Learns the state again, without asking queries, but with the same number of measurements and configurations (but
        here chosen randomly).

        Args:
            epochs: number of epochs.
            period: Period of the data output during the learning.
        """

        # Define all functions needed for the callbacks-----------------------------------------------------------------
        def get_corr(RBM_state, **kwargs):
            statistics = MyCorr().density_statistics(RBM_state, 5000)
            corr = statistics["mean"]
            print(corr)
            target_corr = np.loadtxt("Training_data/tmp/correlator.csv")
            print(target_corr)
            corr_string = ""
            target_corr_string = ""
            for i, c in enumerate(corr):
                corr_string += str(c) + ","
                target_corr_string += str(target_corr[i]) +","
            if os.path.exists("baseline_not_rotated/correlator_full_data.csv"):
                with open("baseline_not_rotated/correlator_full_data.csv", "a") as file_object:
                    file_object.write(corr_string + "\n")
                with open("target_correlator_full_data.csv", "a") as file_object:
                    file_object.write(target_corr_string + "\n")
            else:
                with open("baseline_not_rotated/correlator_full_data.csv", "w") as file_object:
                    file_object.write(corr_string + "\n")
                with open("target_correlator_full_data.csv", "w") as file_object:
                    file_object.write(target_corr_string + "\n")
            if not np.isclose(np.linalg.norm(target_corr),0,rtol=1e-10):
                return np.linalg.norm(corr-target_corr)/np.linalg.norm(target_corr)
            else:
                return np.linalg.norm(corr-target_corr)

        def get_density(RBM_state):
            statistics = MySigmaXSigmaX().density_statistics(RBM_state, 1000)
            r_sxsx = statistics["mean"]
            density = []
            for s in r_sxsx:
                d = 0.5*(1-s)
                density.append(d)
            return np.array(density)

        def get_density_distance(RBM_state, **kwargs):
            density = get_density(RBM_state)
            target_density = np.loadtxt("baseline_not_rotated/target_density.csv")
            density_string = ""
            target_density_string = ""
            for i, d in enumerate(density):
                density_string += str(d) + ","
                target_density_string += str(target_density[i]) +","
            if os.path.exists("baseline_not_rotated/density_full_data.csv"):
                with open("baseline_not_rotated/density_full_data.csv", "a") as file_object:
                    file_object.write(density_string + "\n")
                with open("baseline_not_rotated/target_density_full_data.csv", "a") as file_object:
                    file_object.write(target_density_string + "\n")
            else:
                with open("baseline_not_rotated/density_full_data.csv", "w") as file_object:
                    file_object.write(density_string + "\n")
                with open("baseline_not_rotated/target_density_full_data.csv", "w") as file_object:
                    file_object.write(target_density_string + "\n")
            if not  np.isclose(np.linalg.norm(target_density), 0, rtol=1e-10):
                return np.linalg.norm(density-target_density)/np.linalg.norm(target_density)
            else:
                return np.linalg.norm(density-target_density)

        def get_total_density(RBM_state, **kwargs):
            if self.state.qc_type != "HeisenbergModel":
                density = get_density(RBM_state)
                total_density = 0
                for d in density:
                    total_density += d
            elif self.state.qc_type == "HeisenbergModel":
                new_samples = RBM_state.sample(k=100, num_samples=10000)
                total_density = SigmaZ(absolute=True).statistics_from_samples(RBM_state, new_samples)["mean"]
            return total_density*self.state.qubit_no

        def get_sxsx(RBM_state, **kwargs):
            statistics = MySigmaXSigmaX().density_statistics(RBM_state, 1000)
            r_sxsx = statistics["mean"]
            sxsx = 1/4*np.sum(r_sxsx)
            with open("bl_sxsx_full_data.csv", "a") as file_object:
                file_object.write(", ".join([str(item) for item in r_sxsx]) + "\n")
            return sxsx
        
        def get_sysy(RBM_state, **kwargs):
            statistics = MySigmaYSigmaY().density_statistics(RBM_state, 1000)
            r_sysy = statistics["mean"]
            sysy = 1/4*np.sum(r_sysy)
            with open("bl_sysy_full_data.csv", "a") as file_object:
                file_object.write(", ".join([str(item) for item in r_sysy]) + "\n")
            return sysy
        
        def get_szsz(RBM_state, **kwargs):
            statistics = MySigmaZSigmaZ().density_statistics(RBM_state, 1000)
            r_szsz = statistics["mean"]
            szsz = 1/4*np.sum(r_szsz)
            with open("bl_szsz_full_data.csv", "a") as file_object:
                file_object.write(", ".join([str(item) for item in r_szsz]) + "\n")
            return szsz

        def get_sx(RBM_state, **kwargs):
            sx = SigmaX(absolute=False)
            sx = sx.statistics(RBM_state, num_samples=10000, burn_in=100)["mean"]
            return sx

        def get_sy(RBM_state, **kwargs):
            sy = SigmaY(absolute=False)
            sy = sy.statistics(RBM_state, num_samples=10000, burn_in=100)["mean"]
            return sy

        def get_sz(RBM_state, **kwargs):
            sz = SigmaZ(absolute=False)
            sz = sz.statistics(RBM_state, num_samples=10000, burn_in=100)["mean"]
            return sz

        def get_MarshallSign(RBM_state, **kwargs):
            reference_state = torch.tensor([(-1)**i for i in range(self.state.qubit_no)])
            basis_states = RBM_state.generate_hilbert_space()
            correct_predictions = 0
            for state in basis_states:
                #sign = RBM_state.phase(state)-RBM_state.phase(reference_state)
                sign = np.exp(1j*(RBM_state.phase(state)-RBM_state.phase(reference_state)).item()/2)
                if sign >=0: sign = 1
                else: sign = -1
                n = 0
                for q, qubit in enumerate(state):
                    if q != 0 and q % 2 != 0:
                        if qubit == 1:
                            n += 1
                marshall_sign = (-1)**n
                if marshall_sign == sign:
                    correct_predictions += 1
            return correct_predictions/len(basis_states)

        def get_fidelity(nn_state, target, space=None, **kwargs):
            fid = ts.fidelity(nn_state, target)
            samples = np.loadtxt("baseline_not_rotated/samples.txt")
            qubit_no = len(samples[0])
            fid = fid**(1/qubit_no)
            return fid

        def generateAllBinaryStrings(grouped_samples):
            n = len(list(grouped_samples.keys())[0])
            for i in range(n+1):
                for j, bits in enumerate(itertools.combinations(range(n), i)):
                    s = ['0'] * n
                    for bit in bits:
                        s[bit] = '1'
                    if i == 0 and j == 0:
                        all_binary_strings = [''.join(s)]
                    else:
                        all_binary_strings.append(''.join(s))

            dictionary = {}
            for bit in all_binary_strings:
                dictionary[int(bit)] = bit

            ordered_binary_strings = []
            keys = list(dictionary.keys())
            keys.sort()
            for bit in keys:
                ordered_binary_strings.append(dictionary[bit])
            return ordered_binary_strings

        def get_target_probs(grouped_samples, basis, total_samples_per_basis):
            binary_strings = generateAllBinaryStrings(grouped_samples[basis])
            probs = []
            for sample in binary_strings:
                if sample in list(grouped_samples[basis].keys()):
                    probs.append(grouped_samples[basis][sample]/total_samples_per_basis)
                else:
                    probs.append(0)
            probs = torch.tensor(probs)
            return probs


        def _single_basis_KL(target_probs, nn_probs):
            return torch.sum(target_probs * probs_to_logits(target_probs)) - torch.sum(
                target_probs * probs_to_logits(nn_probs)
            )

        def KL(nn_state, target, space=None, **kwargs):
            train_samples = np.loadtxt("baseline_not_rotated/samples.txt")
            train_bases   = list(np.loadtxt("baseline_not_rotated/train_bases.txt", dtype='str'))
            grouped_samples = {}  # collect the same samples in each basis
            ordered_samples = {}  # get samples in each measurement basis
            bases = []
            samples = []
            for i, basis in enumerate(train_bases):
                string_sample = ""
                string_basis = ""
                for b in basis:
                    string_basis += b
                for s in train_samples[i]:
                    string_sample += str(int(s))
                bases.append(string_basis)
                samples.append(string_sample)
                if string_basis not in list(ordered_samples.keys()):
                    ordered_samples[string_basis] = []
                    grouped_samples[string_basis] = {}
                if string_sample not in list(grouped_samples[string_basis].keys()):
                    grouped_samples[string_basis][string_sample] = 0.0
                ordered_samples[string_basis].append(samples[i])
                grouped_samples[string_basis][string_sample] += 1.0

            space = space if space is not None else nn_state.generate_hilbert_space()
            Z = nn_state.normalization(space)
            KL = 0.0
            if isinstance(nn_state, WaveFunctionBase):
                for basis in set(bases):
                    psi_r = rotate_psi(nn_state, basis, space)
                    nn_probs_r = (cplx.absolute_value(psi_r) ** 2)  / Z
                    nn_probs_r = nn_probs_r.cpu()
                    target_probs_r = get_target_probs(grouped_samples, basis, len(ordered_samples[basis])).cpu()
                    KL += _single_basis_KL(target_probs_r, nn_probs_r)
                KL /= float(len(set(bases)))
            return KL.item()
        #--------------------------------------------------------------------------------------------------------------


        if self.state.state_type == "MPSstate":
                if self.state.qc_type == "LatticeGaugeModel":
                    callbacks = {"KL": KL, "total_density": get_total_density, "density_difference": get_density_distance, "correlator_difference": get_corr}
                elif self.state.qc_type == "HeisenbergModel":
                    callbacks = {"KL": KL, "Sx": get_sx, "Sy": get_sy, "Sz": get_sz, "SxSx":get_sxsx, "SySy":get_sysy, "SzSz": get_szsz, "Marshall": get_MarshallSign}
                else:
                    callbacks = {"KL": KL, "KL(Qucumber)": ts.KL, "rescaled_fidelity": get_fidelity, "total_density": get_total_density,"density_difference": get_density_distance}
        else:
            callbacks = {"rescaled_fidelity": get_fidelity, "KL": ts.KL}
        # Parameters for the RBM: seed and callbacks.
        seed = np.random.randint(1, 100000, 1)[0]


        # Initialize the tomography object.
        tom = tomography.Tomography(self.baseline_path, period, seed)
        # Choose the batchsize equal to the length of the dataset: since phase and amplitude of the
        # RBM state are trained seperately. If not the phase is not learned fast enough.
        batchsize = tom.train_samples.shape[0]
        tom.define_RBM(callbacks, self.state.state_type)

        # Fit the network.

        tom.fit_routine(epochs, batchsize, self.learning_rate, self.k, torch.optim.Adagrad, {},
                        torch.optim.lr_scheduler.MultiStepLR, {"milestones": [1000, 1500], "gamma": 0.75})

        #print(tom.RBMpsi)

        # Plot and save the details of the RBM.
        #tom.plot_callbacks()
        tom.save_details(self.baseline_path)
        return tom.callbacks[0]

    def sample_baseline(self, num_configs, only_total_config_number=False):
        """
        Generates the measurements for the baseline learning.

        Args:
            only_total_config_number: Boolean. If True, only the total number of configurations is measured, i.e. if
                                      self.all_bases = ["ZZ", "XY", "YY", "XY", "XY"], measurements will be performed in
                                      3 randomly chosen bases. If = False, 5 bases will be chosen.
        """

        # Define the parameters for the state.
        # 1. Get all configurations.
        pool_of_configs = self._get_all_configurations(num_configs)
        # 2. Get the string for a configuration with only Zs. Remove Zs from the list of all configurations.
        zz_config = ""
        for i in range(self.state.qubit_no):
            zz_config += "Z"
        pool_of_configs.remove(zz_config)
        # 3. Randomly choose the same number of samples as recently with the active learning procedure. Then add the
        # configuration with only Zs.
        if only_total_config_number == True:
            setting = random.sample(pool_of_configs, len(set(self.all_bases)) - 1)
        else:
            if len(self.all_bases) - 1 <= len(pool_of_configs):
                setting = random.sample(pool_of_configs, len((self.all_bases)) - 1)
            else:
                setting = pool_of_configs
        setting.append(zz_config)
        # 4. Generate the snapshots in these configurations and save them.
        self.state.generate_snapshots(self.n_samples, probability="equal", measurement_setting=setting)
        self.baseline_path = self.folder_name + "/" + str(
            self.n_samples_query) + "samples/" + self.query_type + "/baseline"
        self.state.save_all(self.baseline_path)
        self.state.save_all("../baseline_not_rotated/")

    def plot_callbacks(self):
        """
        Plot and save fidelity, KL and Phase.
        """
        plt.rcParams.update(self.plot_params)
        plt.style.use("seaborn-deep")

        for item in list(self.callbacks.keys()):
            cb = self.callbacks[item]
            fig = plt.figure()
            for RBM_no in list(self.callbacks[item].keys()):
                if item == "Fidelity":
                    y = cb[RBM_no] ** (1 / self.state.qubit_no)
                else:
                    y = cb[RBM_no]
                epoch = np.arange(self.period, self.epochs[RBM_no] + 1, self.period)
                plt.plot(epoch, y, "-o", markeredgecolor="black",
                         label="RBM with seed " + self.RBMseeds[RBM_no])
            if item == "Fidelity":
                plt.ylabel(r"Rescaled Fidelity $f^\frac{1}{N}$")
            else:
                plt.ylabel(item)
            plt.xlabel(r"Learning Steps N")
            plt.legend()
            plt.tight_layout()
            plt.show()
            name = "Training_data/" + self.folder_name + "/" + str(self.n_samples_query) + "samples/" \
                   + self.query_type + "/" + str(self.n_samples) + "samples/"
            if not os.path.exists(name):
                os.makedirs(name)
            fig.savefig(name + item + ".png")
            np.save(name + item, self.callbacks[item])
