# Quantum Simulation Repository: Jaynes–Cummings & Transmon Qubits

This repository contains two Python projects for simulating quantum two-level systems and qubits coupled to resonators, including visualization of the dynamics. The primary goal is to **demonstrate understanding of quantum physics and qubit-resonator dynamics**.

---

## Repository Contents

### 1. **Jaynes–Cummings Model with Dissipation**

- **Main file:** `jaynes_cummings.py`
- **Description:**  
  Simulates the Jaynes–Cummings model: a single qubit coupled to a quantum resonator. Features include:
  - Hamiltonian under the Rotating Wave Approximation (RWA)
  - Optional dissipation:
    - Resonator decay
    - Qubit relaxation
    - Qubit dephasing
  - Time evolution solved via the Lindblad master equation using QuTiP
  - Expectation values computed:
    - Qubit excitation probability
    - Average photon number in the resonator
    - Total system energy
  - Visualization:
    - Rabi oscillations
    - Energy decay over time
- **Code structure:**
  - `JaynesCummings`: defines the physical model and Hamiltonian
  - `Simulation`: handles time evolution
  - `Plotter`: generates visualizations

- **Dependencies:** `qutip`, `numpy`, `matplotlib`

---

### 2. **Transmon Qubit Simulation with Control Pulses**

- **Main file:** `transmon_qubit.py`
- **Description:**  
  Simulates a two-level transmon qubit controlled by a microwave pulse. Features include:
  - Pulse definition (`Pulse`) with Gaussian or rectangular envelopes
  - Numerical calibration of the pulse to achieve a target rotation angle (e.g., π for an X-gate)
  - Qubit Hamiltonian and dynamics (`TransmonQubit`), with the option to use:
    - Rotating Wave Approximation (RWA)
    - Full lab-frame Hamiltonian (non-RWA)
  - Time evolution via numerical integration of the Schrödinger equation (Heun method)
  - Analysis and visualization of results:
    - Populations of |0⟩ and |1⟩ states
    - Bloch sphere trajectory

- **Code structure:**
  - `Pulse`: defines and calibrates control pulses
  - `TransmonQubit`: defines the qubit system
  - `Simulator`: handles time evolution
  - `SimulationResult`: stores and visualizes results

- **Dependencies:** `numpy`, `scipy`, `matplotlib`

---

## Usage Examples

After cloning the repository, ensure all dependencies are installed.

### Jaynes–Cummings
```bash
python jaynes_cummings.py
```
## Outputs
- Rabi oscillations of the qubit
- Average photon number
- Total system energy over time

### Transmon Qubit
```bash
python transmon_simulator.py
```

## Outputs
- Qubit-resonator interactions
- Quantum dissipation
- Dynamics of pulse-controlled qubits

## Repository Goals

- Demonstrate understanding of quantum physics concepts, including:
  - Qubit-resonator interactions
  - Quantum dissipation
  - Dynamics of pulse-controlled qubits
- Provide tools for visualizing and interpreting quantum dynamics

---

## Author / Contribution

**Fabio Calabrese**

Primary contributions: understanding, analysis, and application of physical models; guided code development with AI assistance
