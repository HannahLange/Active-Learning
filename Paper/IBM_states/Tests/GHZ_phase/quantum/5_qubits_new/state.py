#import pyten as p
from qiskit import Aer, execute, IBMQ, QuantumCircuit
from qiskit.providers.ibmq import least_busy
import qiskit.quantum_info as qi
from qiskit.compiler import assemble
import qucumber.utils.data as data
import matplotlib.pyplot as plt
from math import pi
import numpy as np
import os
from itertools import combinations_with_replacement


class IBMstate:
    def __init__(self, IBMtoken, qc_dictionary, simulator_type="classical"):
        """
        Create quantum circuits and prepare different states which can then be measured.
        Args:
            IBMtoken: To use the Quantum Computer an account for IBM Quantum Experience has to be created. If you have
                      done so, your token can be found on https://quantum-computing.ibm.com/account.
            qc_dictionary: dictionary with entries
                           {qc_type: (type of the circuit ("initialize", "1", "0", "GHZ", "Bell3" or "W")),
                           qubit_no: (Number of qubits in the QuantumCircuit. For qc_type = "W" qubit_no = 3, for
                                      qc_type = "Bell3" qubit_no = 2.)
                           init_state: (initialization statevector, ordered as [00..0, 00..1, ..., 11..0, 11..1],
                                        only needed if qc_type = "initialize".)
            simulator_type: "classical" or "quantum". If "classical", simulator is set to
                            Aer.get_backend("aer_simulator").
                            If quantum, the least busy IBM device is chosen which has the appropriate number of qubits.
        """
        self.state_type = "IBMstate"
        self.states = ["initialize", "1", "0", "GHZ", "GHZ_phase","BellSinglet", "BellTriplet", "W", "Heisenberg"]
        self.qc_type = qc_dictionary["qc_type"]
        self.qubit_no = qc_dictionary["qubit_no"]
        self.IBMtoken = IBMtoken
        self.simulator_type = simulator_type
        # All of these variables will be initialized later on:
        self.bases = None
        self.training_bases = None
        self.training_samples = None
        self.circuits = None
        self.simulator = None
        self.device = None
        self.folder_name = None
        self.measurement_no = None
        self.measurement_setting = None

        # set the simulator
        self._get_simulator()

        # initialize the statevector init_state if given.
        init_state = qc_dictionary.get("init_state", None)
        if init_state is not None:
            self.init_state = init_state

        # initialize the quantum circuit specified by qc_type and qubit_no
        self.qc = QuantumCircuit(self.qubit_no, self.qubit_no)
        if self.qc_type == "initialize":
            # Initialize a Quantum Circuit: Qiskit orders thequbits with the
            # most significant bit (MSB) on the left, and the least significant
            # bit (LSB) on the right. For this reason the qubits are swapped
            # after initialization to obtain the right order of the qubits.
            self.qc.initialize(self.init_state)
            for i in range(int(self.qubit_no / 2)):
                self.qc.swap(i, (self.qubit_no - 1 - i))
        if self.qc_type == "1":
            # Circuit with 1-Qubits with variable number of Qubits
            for i in range(self.qubit_no):
                self.qc.x(i)
        if self.qc_type == "0":
            # Circuit with 0-Qubits with variable number of Qubits
            self.qc = self.qc
        if self.qc_type == "GHZ":
            # GHZ State |1...1> + |0...0>
            self.qc.h(0)
            for i in range(self.qubit_no - 1):
                self.qc.cx(i, i + 1)
        if self.qc_type == "GHZ_phase":
            # GHZ State |1...1> + j|0...0>
            self.qc.h(0)
            for i in range(self.qubit_no - 1):
                self.qc.cx(i, i + 1)
            self.qc.s(self.qubit_no-1)
        if self.qc_type == "BellSinglet":
            # Singlet Bell state |10> - |01> with 2 Qubits
            self.qc.h(0)
            self.qc.cx(0, 1)
            self.qc.x(1)
            self.qc.z(1)
        if self.qc_type == "BellTriplet":
            # Triplet Bell state |10> + |01> with 2 Qubits
            self.qc.h(0)
            self.qc.cx(0, 1)
            self.qc.x(1)
        if self.qc_type == "W":
            # W State with 3 Qubits
            if self.qubit_no != 3:
                print("W state can only be used with 3 qubits.")
            self.qc.ry(1.9106332362490184, 0)
            self.qc.cu3(pi / 2, pi / 2, pi / 2, 0, 1)
            self.qc.cx(1, 2)
            self.qc.cx(0, 1)
            self.qc.x(0)
            self.qubit_no = 3

        # add a barrier, only for clarity
        self.qc.barrier()
        # get the respective statevector
        self.target_state = self._get_state()
        print("state vector: "+str(self.target_state))

    def _get_simulator(self):
        """
        Gets the right backend for the simulator specified in self.simulator.
        """
        if self.simulator_type == "classical":
            self.simulator = Aer.get_backend("aer_simulator")
        elif self.simulator_type == "quantum":
            IBMQ.save_account(self.IBMtoken, overwrite=True, filename="/project/th-scratch/h/Hannah.Lange/Projektpraktikum/qiskitrc")
            IBMQ.load_account(filename="/project/th-scratch/h/Hannah.Lange/Projektpraktikum/qiskitrc")
            """If the circuit should run in an IBM Quantum Device: choose a least-busy device with n qubits."""
            provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
            self.device = provider.backends(filters=lambda x: x.configuration().n_qubits >= self.qubit_no
                                                              and not x.configuration().simulator)
            print("use quantum device " + str(least_busy(self.device)))
            self.simulator = provider.get_backend(str(least_busy(self.device)))
        else:
            print("This simulator does not exist. Choose classical or quantum.")


    def _get_state(self):
        """
        Returns the statevector of the prepared QuantumCircuit.
        """
        qc = self.qc.copy()
        for i in range(int(self.qubit_no / 2)):
            qc.swap(i, (self.qubit_no - 1 - i))
        if self.qc_type == "initialize":
            qobj = assemble(qc)
            statevector = Aer.get_backend('statevector_simulator').run(qobj).result().get_statevector()
        else:
            qobj = assemble(qc)
            statevector = qi.Statevector.from_instruction(qc).data
        return statevector


    def generate_training_bases(self, measurement_no, probability):
        """
        Generate training bases (random combinations of X, Y and Z of length = self.qubit_no).
        Args:
            measurement_no: number of measurements
            probability: probability of measuring in X, Y and Z basis."

        Returns:
            self.bases: An array of all bases (each listed once).
            self.training_bases: An array of all training_bases which are measured in self.single_shot or
                                 self.multiple_shots
        """
        training_bases = []
        for m in range(measurement_no):
            basis = []
            basis = list(np.random.choice(list("XYZ"), size=self.qubit_no, p=probability))
            training_bases.append(basis)
        self.training_bases = np.array(training_bases)
        bases = self._gen_bases()
        self.bases = np.array(bases)
        return self.bases, self.training_bases


    def _gen_bases(self):
        """
        Extract a list of all possible training bases out of training_bases.
        """
        bases = []
        for b in self.training_bases:
            i = 0
            for basis in bases:
                if np.array_equal(b, basis):
                    i += 1
            if i == 0:
                bases.append(b)
        self.bases = bases
        return self.bases


    def measure_x(self, qubit, qc):
        """
        Change the basis for measurements in x basis.
        """
        qc.h(qubit)


    def measure_y(self, qubit, qc):
        """
        Change the basis for measurements in y basis.
        """
        qc.sdg(qubit)
        qc.h(qubit)


    def measure_configuration(self, config, shots_per_basis):
        """
        Measure in a basis configuration config (i.e. config = ["X", "Z"]).

        Returns:
            results: The measurement outcomes.
            training_bases: The respective training_bases.
            c: The circuits which were measured.
        """
        c = self.qc.copy()
        for n in range(0, self.qubit_no):
            if config[n] == "X":
                self.measure_x(n, c)
            elif config[n] == "Y":
                self.measure_y(n, c)
        c.measure_all()
        # perform the measurement
        measurement_output = execute(c, backend=self.simulator, shots=shots_per_basis).result().get_counts()
        # append the measurement results to the list results: Qiskit orders the measured bitstring with the
        # most significant bit (MSB) on the left, and the least significant bit (LSB) on the right. For this
        # reason the order of the measurement output has to be reversed.
        res = list(measurement_output.keys())
        results = []
        training_bases = []
        for r in res:
            num = measurement_output[r]
            for n in range(num):
                result = []
                for i in range(self.qubit_no):
                    result.append(int(r[self.qubit_no - 1 - i]))
                results.append(result)
                training_bases.append(config)
        return results, training_bases, c


    def measure_single_shot(self):
        """
        Measure each configuration in self.training_bases is measured one time and save the circuits
        and the results in self.circuits and self.results.
        """
        num_of_shots = 1
        training_samples = []
        circuits = []
        for index, config in enumerate(self.training_bases):
            res, _, c = self.measure_configuration(config, num_of_shots)
            training_samples += res
            circuits.append(c)
            if index % 50 == 0:
                print(str(index) + " samples generated.")
        self.circuits = circuits
        self.training_samples = np.array(training_samples)


    def measure_multiple_shots(self, measurement_no):
        """
        Measure each basis in self.bases int(measurement_no/length of self.bases) times and save the circuits
        and the results in self.circuits and self.results. In this case not self.training_bases generated by
        generate_training_bases is used, but new_training_bases is generated and saved in self.training_bases.
        """
        num_of_shots = int(measurement_no / len(self.bases))
        training_samples = []
        circuits = []
        new_training_bases = []
        i = 0
        for config in self.bases:
            i += 1
            res, training_b, c = self.measure_configuration(config, num_of_shots)
            training_samples += res
            new_training_bases += training_b
            circuits.append(c)
            print(str(num_of_shots * i) + " samples generated.")
        self.circuits = circuits
        self.training_samples = np.array(training_samples)
        self.training_bases = new_training_bases


    def measure_specific_bases(self, measurement_no, specific_bases, probability):
        """
        Measure each basis in self.bases int(measurement_no/length of self.bases) times and save the circuits
        and the results in self.circuits and self.results. In this case not self.training_bases generated by
        generate_training_bases is used, but new_training_bases is generated and saved in self.training_bases.
        """
        training_samples = []
        circuits = []
        new_training_bases = []
        for i, config in enumerate(specific_bases):
            num_of_shots = int(measurement_no * probability[i])
            res, training_b, c = self.measure_configuration(config, num_of_shots)
            training_samples += res
            new_training_bases += training_b
            circuits.append(c)
            print(str(num_of_shots * i) + " samples generated.")
        self.circuits = circuits
        self.training_samples = np.array(training_samples)
        self.training_bases = new_training_bases
        self.bases = specific_bases


    def generate_snapshots(self, measurement_no, probability="equal", measurement_setting="single shots"):
        """
        Generate the training bases and takes measurements in these bases.

        Args:
            measurement_no: number of measurements
            probability: probability of measuring in X, Y and Z basis. Default value is "equal" which translates to
                         [1/3, 1/3, 1/3] for measurement_setting = "single shots" or "multiple shots", and to
                         [x, ..., x] with N*x = 1 for measurement_setting = list of specific bases wih length N.
            measurement_setting: if "single shot", each configuration in self.training_bases is measured one time
                                 (see measure_single_shot). If "multiple shots", each basis in self.bases is measured
                                 int(measurement_no/length of self.bases) times (see measure_multiple_shots).
                                 If it is a vector with basis entries like ["ZZ", "XY"], each basis in the vector is
                                 measured int(measurement_no/length of self.bases) times (see measure_specific_basis).
                                 Default value "single shot".

        Returns:
            self.circuits: The measured_circuits.
            self.training_samples: The measurement outcomes to measurements in self.training_bases.

        """
        self.measurement_no = measurement_no
        self.measurement_setting = measurement_setting
        #state = "[ "
        #for s in self.target_state:
        #    state += str(s) + " "
        #state += "]"
        print("Start measurement of " + self.qc_type + "(" + str(measurement_no) + " configurations).")

        if measurement_setting == "multiple shots":
            if probability == "equal":
                probability = [1 / 3, 1 / 3, 1 / 3]
            self.bases, self.training_bases = self.generate_training_bases(measurement_no, probability)
            self.measure_multiple_shots(measurement_no)
        elif measurement_setting == "single shots":
            if probability == "equal":
                probability = [1 / 3, 1 / 3, 1 / 3]
            self.bases, self.training_bases = self.generate_training_bases(measurement_no, probability)
            self.measure_single_shot()
        else:
            if probability == "equal":
                probability = list(np.ones(len(measurement_setting)) / len(measurement_setting))
            self.measure_specific_bases(measurement_no, measurement_setting, probability)
        return self.circuits, self.training_samples


    def save_all(self, folder_name):
        """
        Save self.bases in folder_name/bases.txt, self.training_bases in folder_name/train_bases.txt,
        self.training_bases in folder_name/samples and self.target_state in folder_name/psi.txt.

        Args:
            folder_name: Path to the folder where everything is saved.
        """
        self.folder_name = "Training_data/" + folder_name
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        bases_path = self.folder_name + "/bases.txt"
        train_bases_path = self.folder_name + "/train_bases.txt"
        train_path = self.folder_name + "/samples.txt"
        psi_path = self.folder_name + "/psi.txt"

        file0 = open(bases_path, "w")
        for b in self.bases:
            string = str(b[0])
            for j in range(1, len(b)):
                string += str(b[j])
            string += " \n"
            file0.write(string)
        file0.close()

        file1 = open(train_bases_path, "w")
        for b in self.training_bases:
            string = str(b[0])
            for j in range(1, len(b)):
                string += " " + str(b[j])
            string += " \n"
            file1.write(string)
        file1.close()

        file2 = open(train_path, "w")
        for sa in self.training_samples:
            string = str(sa[0])
            for j in range(1, len(sa)):
                string += " " + str(sa[j])
            string += " \n"
            file2.write(string)
        file2.close()

        file3 = open(psi_path, "w")
        for p in list(self._get_state()):
            string = str(p.real) + " " + str(p.imag) + " \n"
            file3.write(string)
        file3.close()
        print(
            "Bases, training bases, training samples and target statevector are saved in the folder: " + self.folder_name)


    def load_data(self):
        """
        Load the data which was saved in save_all(). Needed to generate the histogramm in visualize_data().
        """
        bases_path = self.folder_name + "/bases.txt"
        train_bases_path = self.folder_name + "/train_bases.txt"
        train_path = self.folder_name + "/samples.txt"
        psi_path = self.folder_name + "/psi.txt"

        train_samples, true_psi, train_bases, bases = data.load_data(train_path, psi_path, train_bases_path, bases_path)
        return train_samples, true_psi, train_bases, bases


    def normalize_dictionary(self, dictionary):
        """
        Normalize the dictionary of the measurements generated in generate_dictionary().

        Args:
            dictionary: Dictionary with the values which are normalized to 1.
        Returns:
            normalized_dictionary: Dictionary with normalized values.
        """
        normalized_dictionary = {}
        for key in list(dictionary.keys()):
            no = 0
            normalized_dictionary[key] = {}
            for k in list(dictionary[key].keys()):
                no += dictionary[key][k]
            # print(str(key)+": "+str(no))
            for k in list(dictionary[key].keys()):
                if no == 0:
                    normalized_dictionary[key][k] = 0
                else:
                    normalized_dictionary[key][k] = dictionary[key][k] / no
        return normalized_dictionary


    def generate_dictionary(self):
        """
        Generate a dictionary of all measurements. Needed to generate the histogram in visualize_data().

        Returns:
            dictionary: The generated dictionary.
        """
        train_samples, true_psi, train_bases, bases = self.load_data()

        statevector_basis_combinations = {}
        for statevector_basis in list(combinations_with_replacement([0, 1], self.qubit_no)):
            statevector_basis_combinations[str(statevector_basis)] = 0

        dictionary = {}
        for b in bases:
            label = "("
            for i in b:
                label += i
            label += ")"
            dictionary[label] = statevector_basis_combinations
            for j in range(len(train_bases)):
                tb = train_bases[j][0]
                for t in train_bases[j][1:]:
                    tb += t
                if (tb == b):
                    label2 = []
                    for number in train_samples[j]:
                        label2.append(int(number))
                    label2 = str(tuple(label2))
                    if label2 in list(dictionary[label].keys()):
                        dictionary[label][label2] += 1
        dictionary = self.normalize_dictionary(dictionary)
        return dictionary


    def visualize_data(self):
        """
        Generate a histogram plot for the measurement outcomes of each basis configuration.
        """
        dictionary = self.generate_dictionary()
        print(dictionary)

        fig, ax = plt.subplots(len(self.bases), figsize=(20, len(self.bases) * 5))
        i = 0
        for key in dictionary:
            ax[i].bar(dictionary[key].keys(), dictionary[key].values())
            ax[i].set_title(str(key))
            i += 1
        plt.show()
        plt.tight_layout()
        fig.savefig(self.folder_name + "/measurements.png", dpi = 1200)



class MPSstate(IBMstate):
    def __init__(self, qc_dictionary):
        super().__init__(None, qc_dictionary)
        self.state_type = "MPSstate"
        self.qc_dictionary = qc_dictionary
        self.states = ["GHZ", "LatticeGaugeModel"]
        self.calc_observables = False
        self.density = None
        self.sigmax = None
        self.correlator = None

        if self.qc_type == "GHZ":
            self.lattice = p.mp.lat.snap.genSpinHalfLattice(self.qubit_no, "nil")
            up = p.mp.generateNearVacuumState(self.lattice)
            dn = p.mp.generateNearVacuumState(self.lattice)
            for i in range(self.qubit_no):
                p.mp.apply_op_naive(up, self.lattice.get("pszup", i))
                up.normalise()
                p.mp.apply_op_naive(dn, self.lattice.get("pszdn", i))
                dn.normalise()
            self.qc = up + dn
            self.qc.normalise()

        if self.qc_type == "LatticeGaugeModel":
            self.lattice = self.gen_lgtm()
            rnd_mps = p.mp.genSampledState(self.lattice)
            opt_states = self.dmrg(rnd_mps)
            self.qc = opt_states[-1]

    def gen_lgtm(self):
        """
        Helper function, setting up the LGT model
        """
        # Parameters
        h = self.qc_dictionary["Parameters"]["h"]
        lamb = self.qc_dictionary["Parameters"]["lamb"]
        mu = self.qc_dictionary["Parameters"]["mu"]
        if mu == 0:
            mu = 0.0000001
        # lattice
        lat = p.mp.lat.snap.genSpinHalfLattice(self.qubit_no,"nil")
        # hamiltonian
        ham = []
        for i in range(1, self.qubit_no-1): 
            ham += [-4 * lamb * lat.get("sx",i-1) * lat.get("sz", i) * lat.get("sx",i+1) ]
            ham += [lamb*lat.get("sz",i) ]
        for i in range(0, self.qubit_no-1):
            ham += [ +mu * lat.get("sx",i+1) * lat.get("sx",i) ]
        for i in range(0, self.qubit_no):
            ham += [ -2 * h * lat.get("sx",i) ]
        ham = p.mp.addLog(ham)
        lat.add("H","H",ham)
        return lat

    def dmrg(self, state, stage_desc = ["(m 50 x 20)","(m 100 x 20)"]):
        """
        Helper function, running DMRG
        """
        dmrgconf = p.dmrg.DMRGConfig()
        for stage in stage_desc:
            dmrgconf.stages += [p.dmrg.DMRGStage(stage)]
        pdmrg = p.mp.dmrg.PDMRG(state, [self.lattice.get("H")], dmrgconf)
        opt_states = []
        for stage in stage_desc:
            mps = pdmrg.run()
            opt_states.append(mps)
        return opt_states

    def measure_configuration(self, config, shots_per_basis):
        """
        Measure in a basis configuration config (i.e. config = ["X", "Z"]).

        Returns:
            results: The measurement outcomes.
            training_bases: The respective training_bases.
            c: The circuits which were measured.
        """
        measurement_config = []
        for i in range(0, self.qubit_no):
            if config[i] == "X":
                measurement_config.append([self.lattice.get("psxdn",i), self.lattice.get("psxup",i)])
            elif config[i] == "Y":
                measurement_config.append([self.lattice.get("psydn",i), self.lattice.get("psyup",i)])
            elif config[i] == "Z":
                measurement_config.append([self.lattice.get("pszdn",i), self.lattice.get("pszup",i)])
        # perform the measurement
        rndseed = 4
        results = p.mp.snap.shots(self.lattice, self.qc, shots_per_basis, measurement_config, rndseed)
        training_bases = []
        for r in results:
            training_bases.append(config)
        c = None
        return results, training_bases, c

    def get_observables(self):
        self.calc_observables = True
        # calculate the density
        self.density = []
        sxsx = []
        self.total_density = 0
        for i in range(self.qubit_no-1):
            sxsx_i = p.mp.expectation(self.qc, self.lattice.get("sx", i) * self.lattice.get("sx", i+1))
            sxsx.append(sxsx_i)
            density = 0.5*(1-4*sxsx_i)
            self.total_density += density
            self.density.append(density)
        assert np.all(np.isclose(np.imag(self.density),0))
        self.density = np.real(self.density)

        # calculate sigmax
        self.sigmax = 0.0
        for i in range(self.qubit_no):
            self.sigmax += p.mp.expectation(self.qc, self.lattice.get("sx", i))
        self.sigmax = np.real(self.sigmax)

        #calculate the correlator
        self.correlator = []
        n_i = 0.5*self.lattice.get("I")+(-1)*0.5*4*self.lattice.get("sx", int(self.qubit_no/2)-1) * self.lattice.get("sx", int(self.qubit_no/2))
        for qubit_j in range(int(self.qubit_no/2),self.qubit_no-1):
            n_j = 0.5*self.lattice.get("I")+(-1)*0.5*4*self.lattice.get("sx", qubit_j) * self.lattice.get("sx", qubit_j+1)
            corr = n_j
            z_product = 2*self.lattice.get("sz",int(self.qubit_no/2))
            for l in range(int(self.qubit_no/2)+1,qubit_j+1):
                z_product *= 2*self.lattice.get("sz",l)
            corr *= z_product
            corr *= n_i
            correlator = p.mp.expectation(self.qc, corr)
            assert np.all(np.isclose(np.imag(correlator),0))
            self.correlator.append(np.real(correlator))
        return sxsx, self.density, self.total_density, self.sigmax, self.correlator

    def save_all(self, folder_name):
        """
        Save self.bases in folder_name/bases.txt, self.training_bases in folder_name/train_bases.txt,
        self.training_bases in folder_name/samples and self.target_state in folder_name/psi.txt.

        Args:
            folder_name: Path to the folder where everything is saved.
        """
        self.folder_name = "Training_data/" + folder_name
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        bases_path = self.folder_name + "/bases.txt"
        train_bases_path = self.folder_name + "/train_bases.txt"
        train_path = self.folder_name + "/samples.txt"
        psi_path = self.folder_name + "/psi.txt"

        file0 = open(bases_path, "w")
        for b in self.bases:
            string = str(b[0])
            for j in range(1, len(b)):
                string += str(b[j])
            string += " \n"
            file0.write(string)
        file0.close()

        file1 = open(train_bases_path, "w")
        for b in self.training_bases:
            string = str(b[0])
            for j in range(1, len(b)):
                string += " " + str(b[j])
            string += " \n"
            file1.write(string)
        file1.close()

        file2 = open(train_path, "w")
        for sa in self.training_samples:
            string = str(sa[0])
            for j in range(1, len(sa)):
                string += " " + str(sa[j])
            string += " \n"
            file2.write(string)
        file2.close()

        if self.qc_type == "GHZ":
            file3 = open(psi_path, "w")
            for p in range(2**self.qubit_no):
                if p == 0 or p == (2**self.qubit_no-1): 
                    string = str(np.sqrt(1/2)) + " " + str(0.0) + "\n"
                else: 
                    string = str(0.0) + " " + str(0.0) + "\n"
                file3.write(string)
            file3.close()

        if self.calc_observables:
            obs_path = self.folder_name + "/target_density.csv"
            np.savetxt(obs_path, self.density, delimiter = ",")
            obs_path = self.folder_name + "/target_total_density.csv"
            np.savetxt(obs_path, [self.total_density], delimiter = ",")
            obs_path = self.folder_name + "/target_sigmax.csv"
            np.savetxt(obs_path, [self.sigmax], delimiter = "")
            obs_path = self.folder_name + "/correlator.csv"
            np.savetxt(obs_path, self.correlator, delimiter = ",")
        print(
            "Bases, training bases and training samples are saved in the folder: " + self.folder_name)

