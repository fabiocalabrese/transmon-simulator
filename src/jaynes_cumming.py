"""
Jaynes-Cummings with optional counter-rotating terms (no-RWA) and dissipation.
Structure: JaynesCummings (model), Simulation (execution), Plotter (visualization).

Requires: qutip, numpy, matplotlib
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import qutip as qt

matplotlib.use("TkAgg")

# -----------------------------
# Class for the Jaynes-Cummings Model
# -----------------------------
class JaynesCummings:
    """
    This class defines the physical model. It sets up the Hilbert space
    and constructs the system's Hamiltonian based on the provided physical parameters.
    """
    def __init__(self, wq, wr, g, n_levels, kappa=0.0, gamma_relax=0.0, gamma_deph=0.0):
        """
        wq          : qubit frequency (rad/s)
        wr          : resonator frequency (rad/s)
        g           : coupling strength (rad/s)
        n_levels    : truncation level of the resonator's Hilbert space (Fock states)
        kappa       : resonator loss rate (s^-1)
        gamma_relax : qubit relaxation rate (T1) (s^-1)
        gamma_deph  : pure dephasing rate for the qubit (s^-1)
        """
        # Store physical parameters
        self.wq = wq
        self.wr = wr
        self.g = g
        self.n_levels = n_levels

        # Store dissipation rates
        self.kappa = kappa
        self.gamma_relax = gamma_relax
        self.gamma_deph = gamma_deph

        # --- Define quantum operators for the resonator (field) ---
        # Annihilation operator
        self.a = qt.destroy(n_levels)
        # Creation operator
        self.adag = self.a.dag()
        # Photon number operator
        self.n_r = qt.num(n_levels)

        # --- Define quantum operators for the qubit ---
        # Convention: |g> = [1,0]^T, |e> = [0,1]^T
        # sigma+ = |e><g|, sigma- = |g><e|
        # Raising operator: transitions from |g> to |e>
        self.sp = qt.Qobj(np.array([[0, 0], [1, 0]]))   # |e><g|
        # Lowering operator: transitions from |e> to |g>
        self.sm = qt.Qobj(np.array([[0, 1], [0, 0]]))   # |g><e|
        # Pauli-Z operator
        self.sz = qt.sigmaz()

        # Build the total Hamiltonian for the combined system
        self.H = self.build_hamiltonian()

    def build_hamiltonian(self):
        """
        Constructs the total Hamiltonian for the qubit-resonator system in the
        combined (tensor product) Hilbert space.
        """
        # Local terms (non-interacting parts)
        # Qubit Hamiltonian: H_q = -0.5 * wq * (sz ⊗ I)
        # The tensor product with the identity `qeye` extends the qubit operator to the full Hilbert space.
        H_q = -0.5 * self.wq * qt.tensor(self.sz, qt.qeye(self.n_levels))
        # Resonator Hamiltonian: H_r = wr * (I ⊗ a†a)
        H_r = self.wr * qt.tensor(qt.qeye(2), self.n_r)

        # Interaction term under the Rotating Wave Approximation (RWA)
        # H_int = g * (σ+ ⊗ a + σ- ⊗ a†)
        H_int_rwa = self.g * (qt.tensor(self.sp, self.a) + qt.tensor(self.sm, self.adag))

        # The total Hamiltonian is the sum of all parts
        return H_q + H_r + H_int_rwa

# -----------------------------
# Class for the simulation (using the master equation if collapse operators are provided)
# -----------------------------
class Simulation:
    """
    This class handles the execution of the time evolution. It takes the model,
    an initial state, and a time list, and then solves the master equation.
    """
    def __init__(self, model, psi0, tlist):
        """
        model        : An instance of the JaynesCummings class
        psi0         : The initial state of the system (a ket Qobj)
        tlist        : A numpy array of time points for the simulation
        """
        self.model = model
        self.psi0 = psi0
        self.tlist = tlist
        self.result = None

    def build_c_ops(self):
        """
        Constructs the list of collapse operators (c_ops) that describe the
        system's interaction with its environment (dissipation).
        These are used by the Lindblad master equation solver in QuTiP.
        """
        c_ops = []
        # Resonator decay: sqrt(kappa) * (I ⊗ a)
        # This operator describes the loss of a single photon from the resonator.
        if self.model.kappa > 0:
            c_ops.append(np.sqrt(self.model.kappa) * qt.tensor(qt.qeye(2), self.model.a))
        # Qubit relaxation: sqrt(gamma_relax) * (σ- ⊗ I)
        # This operator describes the qubit decaying from |e> to |g>.
        if self.model.gamma_relax > 0:
            c_ops.append(np.sqrt(self.model.gamma_relax) * qt.tensor(self.model.sm, qt.qeye(self.model.n_levels)))
        # Pure dephasing: sqrt(gamma_deph) * (σz ⊗ I)
        # This operator describes the loss of phase information without energy change.
        if self.model.gamma_deph > 0:
            c_ops.append(np.sqrt(self.model.gamma_deph) * qt.tensor(self.model.sz, qt.qeye(self.model.n_levels)))
        return c_ops

    def run(self, e_ops=None, store_states=True):
        """
        Runs the simulation using qutip.mesolve.
        If e_ops (expectation operators) are not provided, it defaults to calculating:
          - The projector onto the qubit's excited state (σ+σ- ⊗ I)
          - The average number of photons (I ⊗ a†a)
          - The total energy <H>
        """
        if e_ops is None:
            # Projector for qubit excited state |e><e|
            qubit_proj_e = qt.tensor(self.model.sp * self.model.sm, qt.qeye(self.model.n_levels))
            # Photon number operator for the resonator
            resonator_n = qt.tensor(qt.qeye(2), self.model.n_r)
            # Total energy operator
            energy_op = self.model.H
            e_ops = [qubit_proj_e, resonator_n, energy_op]

        # Get the collapse operators for dissipation
        c_ops = self.build_c_ops()
        # Solve the master equation: H, psi0, tlist, c_ops, e_ops
        self.result = qt.mesolve(self.model.H, self.psi0, self.tlist, c_ops, e_ops, options=qt.Options(store_states=store_states))
        return self.result



# -----------------------------
# Plotter Class
# -----------------------------
class Plotter:
    """
    This class is responsible for visualizing the simulation results.
    """
    def __init__(self, sim_result, tlist, model=None, qubit_entropy=None):
        self.result = sim_result
        self.tlist = tlist
        self.model = model
        self.qubit_entropy = qubit_entropy

    def plot_oscillations_and_energy(self):
        """
        Plots the qubit excitation probability, the average photon number,
        and the total system energy as a function of time.
        """
        # Extract expectation values from the result object
        # expect[0]=P_e, expect[1]=<n>, expect[2]=<H>
        p_excited = self.result.expect[0]
        n_photons = self.result.expect[1]
        energy = self.result.expect[2] if len(self.result.expect) > 2 else None

        # Convert time to nanoseconds for better readability
        t_ns = self.tlist * 1e9

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # Subplot 1: Qubit and resonator dynamics (Rabi oscillations)
        ax1.plot(t_ns, p_excited, label="Qubit Probability in |e⟩", linewidth=2)
        ax1.plot(t_ns, n_photons, label="Average Photon Number ⟨n⟩", linestyle='--', linewidth=2)
        ax1.set_ylabel("Expectation Value")
        ax1.set_title("Jaynes–Cummings (RWA, κ={}, γ1={}, γφ={})"
                      .format(
                              getattr(self.model, 'kappa', 0),
                              getattr(self.model, 'gamma_relax', 0),
                              getattr(self.model, 'gamma_deph', 0)))
        ax1.grid(True, linestyle=':')
        ax1.legend(fontsize=10)
        ax1.set_ylim([-0.05, 1.05])

        # Subplot 2: Total energy
        # In a closed system (no dissipation), this would be a flat line.
        # With dissipation, the total energy decreases over time.
        if energy is not None:
            ax2.plot(t_ns, energy, label="Total Energy ⟨H⟩", color='orange', linewidth=1.5)
            ax2.set_xlabel("Time (ns)")
            ax2.set_ylabel("Energy [rad/s]")
            ax2.grid(True, linestyle=':')
            ax2.legend(fontsize=10)
        else:
            ax2.set_visible(False)

        plt.tight_layout()
        plt.show()


# -----------------------------
# EXAMPLE USAGE
# -----------------------------
if __name__ == "__main__":
    # --- Define physical parameters for the simulation ---
    # Frequencies are defined in Hz and converted to angular frequency (rad/s)
    wq = 5.0 * 2 * np.pi * 1e9    # Qubit frequency: 5 GHz
    wr = 5.0 * 2 * np.pi * 1e9    # Resonator frequency: 5 GHz (on resonance)
    g = 0.05 * 2 * np.pi * 1e9   # Coupling strength: 50 MHz

    # Truncate the resonator's Hilbert space to N levels (0 to N-1 photons)
    N = 20
    # Define a time list for the simulation, scaled by the Rabi period (pi/g)
    t_period = np.pi / g
    tlist = np.linspace(0, 4 * t_period, 1001)

    # --- Define realistic dissipation rates (in s^-1) ---
    kappa = 1e6       # Resonator decay rate (1 MHz)
    gamma_relax = 5e5 # Qubit T1 decay rate (0.5 MHz)
    gamma_deph = 1e5  # Qubit pure dephasing rate (0.1 MHz)

    # --- Step 1: Build the model ---
    jc = JaynesCummings(wq, wr, g, N, kappa=kappa, gamma_relax=gamma_relax, gamma_deph=gamma_deph)

    # --- Step 2: Define the initial state ---
    # Qubit in the excited state |e⟩, resonator field in the vacuum state |0⟩
    psi0 = qt.tensor(qt.basis(2, 1), qt.fock(N, 0))

    # --- Step 3: Run the simulation ---
    sim = Simulation(jc, psi0, tlist)
    # Run using the default expectation operators (P_e, <n>, <H>)
    result = sim.run()

    # --- Step 4: Plot the results ---
    plotter = Plotter(result, tlist, model=jc)
    plotter.plot_oscillations_and_energy()