import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root_scalar
from mpl_toolkits.mplot3d import Axes3D

# Use a backend that works well in different environments
matplotlib.use("TkAgg")

# --- Constants (Pauli Matrices) ---
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)


class Pulse:
    """
    Represents the microwave control pulse applied to the qubit.

    This class is responsible for defining the pulse's shape (envelope),
    its carrier signal, and calibrating its parameters to achieve a
    specific quantum gate rotation.
    """

    def __init__(self, envelope_type="gaussian", V0=1.0, wd=0.0, phi=0.0, mu=0.0, sigma=1.0):
        self.envelope_type = envelope_type
        self.V0 = V0  # Drive amplitude (rad/s for RWA)
        self.wd = wd  # Drive frequency (rad/s)
        self.phi = phi
        self.mu = mu
        self.sigma = sigma

    def envelope(self, t):
        """Calculates the pulse envelope s(t) at a given time t."""
        if self.envelope_type == "gaussian":
            # Avoid division by zero if sigma is not set
            if self.sigma == 0:
                return np.zeros_like(t)
            return np.exp(-0.5 * ((t - self.mu) / self.sigma) ** 2)
        elif self.envelope_type == "rectangular":
            # Defines a rectangular pulse of duration 'sigma'
            return np.where((t >= self.mu) & (t <= self.mu + self.sigma), 1.0, 0.0)
        else:
            raise ValueError(f"Envelope type '{self.envelope_type}' not recognized.")

    def signal(self, t):
        """Calculates the full drive voltage V(t) = V0 * s(t) * sin(wd*t + phi)."""
        return self.V0 * self.envelope(t) * np.sin(self.wd * t + self.phi)

    def calibrate_for_rotation(self, theta, t_range):
        """
        Numerically finds the gaussian `sigma` required to produce a total
        pulse area corresponding to a rotation angle `theta`.
        The area is V0 * integral(envelope(t) dt).
        """
        if self.envelope_type != "gaussian":
            raise ValueError("Calibration is only implemented for Gaussian pulses.")

        t_min, t_max = t_range

        # Objective function for the solver: Area(sigma) - theta = 0
        def objective(sigma_val):
            # Temporarily set sigma to the value being tested by the solver
            self.sigma = sigma_val
            # The area is the integral of the envelope
            integral_val, _ = quad(self.envelope, t_min, t_max)
            # The total pulse area is V0 * integral
            pulse_area = self.V0 * integral_val
            return pulse_area - theta

        # Find the root of the objective function
        try:
            sol = root_scalar(objective, bracket=[1e-12, 1e-6], method='brentq')
            if not sol.converged:
                raise RuntimeError("Sigma calibration failed to converge.")
            self.sigma = sol.root
            return self.sigma
        except ValueError:
            raise RuntimeError(
                "Failed to find a valid bracket for the root solver. "
                "The target angle might be too large for the given V0 and time range."
            )

    def plot(self, tlist):
        """Plots the pulse envelope and the full drive signal."""
        s_t = self.envelope(tlist)
        V_t = self.signal(tlist)

        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig.suptitle("Control Pulse Waveforms", fontsize=14)

        ax[0].plot(tlist * 1e9, s_t, color='tab:blue', label=f"σ = {self.sigma * 1e9:.2f} ns")
        ax[0].set_ylabel("Envelope s(t)")
        ax[0].set_title(f"Envelope ({self.envelope_type.capitalize()})")
        ax[0].grid(True)
        ax[0].legend()

        ax[1].plot(tlist * 1e9, V_t, color='tab:orange')
        ax[1].set_xlabel("Time (ns)")
        ax[1].set_ylabel("Drive V(t)")
        ax[1].set_title("Full Drive Signal")
        ax[1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig


class TransmonQubit:
    """
    Represents the two-level transmon system.

    This class defines the physics of the qubit, including its natural
    frequency and how it interacts with a control pulse via its Hamiltonian.
    """

    def __init__(self, wq):
        self.wq = wq  # Qubit frequency (rad/s)

    def get_hamiltonian(self, t, pulse, rotating_frame=False):
        """Constructs the total Hamiltonian H(t) for a given pulse."""
        # Unperturbed Hamiltonian (H0)
        if not rotating_frame:
            H0 = -0.5 * self.wq * SIGMA_Z
        else:
            # In rotating frame, H0 includes the detuning
            Delta = self.wq - pulse.wd
            H0 = -0.5 * Delta * SIGMA_Z

        # Drive Hamiltonian (Hd)
        if not rotating_frame:
            # Lab frame: Hd is driven by the full oscillating signal
            Hd = pulse.signal(t) * SIGMA_Y
        else:
            # Rotating frame (RWA): Hd is driven by the slow envelope
            s = pulse.envelope(t)
            I = np.cos(pulse.phi)
            Q = np.sin(pulse.phi)
            Hd = -0.5 * pulse.V0 * s * (I * SIGMA_X + Q * SIGMA_Y)

        return H0 + Hd


class Simulator:
    """
    Orchestrates the time-evolution simulation.

    This class takes a physical system (qubit) and a control pulse,
    and solves the Schrödinger equation to find the state of the system
    over time.
    """

    def __init__(self, qubit, pulse, rotating_frame=False):
        self.qubit = qubit
        self.pulse = pulse
        self.rotating_frame = rotating_frame

    def _schrodinger_rhs(self, t, psi):
        """Right-hand side of the Schrödinger eq: d(psi)/dt = -i * H(t) * psi."""
        H = self.qubit.get_hamiltonian(t, self.pulse, self.rotating_frame)
        return -1j * (H @ psi)

    def run(self, psi0, tlist):
        """
        Evolves the initial state `psi0` over the time steps `tlist`
        using the Heun method (a second-order Runge-Kutta method).
        """
        dt = tlist[1] - tlist[0]
        psi_history = np.zeros((len(tlist), len(psi0)), dtype=complex)
        psi_history[0] = psi0 / np.linalg.norm(psi0)  # Normalize initial state

        for i in range(len(tlist) - 1):
            psi_current = psi_history[i]
            t = tlist[i]

            # Heun's method steps
            k1 = self._schrodinger_rhs(t, psi_current)
            psi_predictor = psi_current + dt * k1
            k2 = self._schrodinger_rhs(t + dt, psi_predictor)

            psi_next = psi_current + (dt / 2.0) * (k1 + k2)

            # Normalize state at each step to conserve probability
            psi_history[i + 1] = psi_next / np.linalg.norm(psi_next)

        return SimulationResult(tlist, psi_history)


class SimulationResult:
    """
    Stores and visualizes the results from a simulation run.
    """

    def __init__(self, tlist, psi_history):
        self.tlist = tlist
        self.psi_history = psi_history
        self.psi_final = psi_history[-1]

    def plot_populations(self):
        """Plots the population of the |0> and |1> states over time."""
        p0 = np.abs(self.psi_history[:, 0]) ** 2
        p1 = np.abs(self.psi_history[:, 1]) ** 2

        fig = plt.figure(figsize=(8, 5))
        plt.plot(self.tlist * 1e9, p0, label="P(|0⟩)", color='tab:blue')
        plt.plot(self.tlist * 1e9, p1, label="P(|1⟩)", color='tab:red')
        plt.xlabel("Time (ns)")
        plt.ylabel("Population")
        plt.title("Qubit State Population vs. Time")
        plt.legend()
        plt.grid(True)
        plt.ylim([-0.05, 1.05])
        return fig

    def plot_bloch_trajectory(self):
        """Plots the trajectory of the state vector on the Bloch sphere."""
        # Calculate Bloch vector components for each time step
        sx = np.einsum('ij,ji->i', self.psi_history.conj(), SIGMA_X @ self.psi_history.T).real
        sy = np.einsum('ij,ji->i', self.psi_history.conj(), SIGMA_Y @ self.psi_history.T).real
        sz = np.einsum('ij,ji->i', self.psi_history.conj(), SIGMA_Z @ self.psi_history.T).real

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Draw the wireframe sphere
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:30j]
        x_sphere = np.cos(u) * np.sin(v)
        y_sphere = np.sin(u) * np.sin(v)
        z_sphere = np.cos(v)
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.2)

        # Plot trajectory
        ax.plot(sx, sy, sz, color='tab:purple', lw=2.5, label='Trajectory')

        # Mark start (red circle) and end (green star) points
        ax.scatter(sx[0], sy[0], sz[0], color='red', s=150, label='Start |0⟩', marker='o')
        ax.scatter(sx[-1], sy[-1], sz[-1], color='green', s=250, label='End State', marker='*')

        ax.set_xlabel("⟨σx⟩")
        ax.set_ylabel("⟨σy⟩")
        ax.set_zlabel("⟨σz⟩")
        ax.set_title("Trajectory on Bloch Sphere")
        ax.legend()
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio for a perfect sphere
        return fig


# =============================================================================
# Main script execution
# =============================================================================
if __name__ == "__main__":

    # --- 1. Define Physical Parameters ---
    wq = 2 * np.pi * 5.0e8  # Qubit frequency: 500 MHz
    wd = 2 * np.pi * 5.0e8  # Drive frequency: 500 MHz (on-resonance)
    V0 = 1.0e8  # Drive amplitude (in rad/s for RWA context)
    phi = np.pi / 2  # Drive phase (pi/2 for rotation around X-axis in RWA)
    mu = 25e-9  # Center of the Gaussian pulse (25 ns)
    theta_gate = np.pi  # Target rotation angle: pi for an X-gate (NOT gate)
    t_start, t_end = 0, 50e-9  # Simulation time window

    # Time steps for the simulation
    tlist = np.linspace(t_start, t_end, 2001)

    # Initial state of the qubit: |ψ(0)⟩ = |0⟩
    psi0 = np.array([1, 0], dtype=complex)

    # --- 2. Setup the Simulation Components ---

    # Create a TransmonQubit instance
    qubit = TransmonQubit(wq=wq)

    # Create a Pulse instance (sigma is unknown for now)
    # We are simulating in the lab frame (non-RWA)
    USE_ROTATING_FRAME = True

    if USE_ROTATING_FRAME:
        # In RWA, V0 directly corresponds to Rabi frequency Omega.
        # phi=pi/2 -> sigma_y drive, which causes rotation around X-axis.
        pulse = Pulse(envelope_type="gaussian", V0=V0, wd=wd, phi=phi, mu=mu)
    else:
        # In Lab frame, V0 is a voltage amplitude.
        # We need to scale it appropriately, but for this example we keep it.
        # phi=pi/2 means the drive is a cosine.
        pulse = Pulse(envelope_type="gaussian", V0=V0, wd=wd, phi=phi, mu=mu)

    # Calibrate the pulse to achieve the desired rotation
    if pulse.envelope_type == "gaussian":
        calculated_sigma = pulse.calibrate_for_rotation(theta_gate, (t_start, t_end))
        print(f"Calibrated Gaussian sigma for a π-pulse: {calculated_sigma * 1e9:.3f} ns")

    # Create the Simulator instance
    simulator = Simulator(qubit, pulse, rotating_frame=USE_ROTATING_FRAME)

    # --- 3. Run the Simulation ---
    print("Running simulation...")
    results = simulator.run(psi0, tlist)
    print("Simulation finished.")
    print(f"Initial state: |ψ(0)⟩ = {np.round(results.psi_history[0], 2)}")
    print(f"Final state:   |ψ(t)⟩ = {np.round(results.psi_final, 2)}")

    # --- 4. Visualize the Results ---
    fig1 = pulse.plot(tlist)
    fig2 = results.plot_populations()
    fig3 = results.plot_bloch_trajectory()

    plt.show()