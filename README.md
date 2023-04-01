# Adaptive Quantum State Tomography with Active Learning
Implementation of Active Learning (AL) Quantum State Tomography for various quantum states generated on IBMs quantum devices or with DMRG. Our AL scheme decides, based on previous measurements, which meansurement configuration should be considered in the next measurement. In order to find the measurement configuration that will improve the learned state representation most efficiently, we apply the \textit{query-by-committee} strategy, where the members of the committee are represented by restricted Boltzmann machine (RBM) quantum states. See our paper for further details: https://arxiv.org/abs/2203.15719 .

# Requirements:
QuCumber for the RBM reconstruction: https://qucumber.readthedocs.io/en/stable/

An example can be run by calling example.py.
