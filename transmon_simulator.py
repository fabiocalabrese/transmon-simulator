import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root_scalar
matplotlib.use("TkAgg")

class TransmonSimulator:
    def __init__(self, wq, wd, V0, phi=0.0, envelope_type="gaussian",
                 mu=0.0, sigma=3.0, t0=0.0):
        """
        wq: frequenza qubit (rad/s)
        wd: frequenza drive (rad/s)
        V0: ampiezza drive
        phi: fase drive
        envelope_type: 'gaussian' o 'rectangular'
        mu, sigma: parametri dell'envelope (se gaussiana)
        t0: inizio impulso
        """
        self.wq = wq
        self.wd = wd
        self.V0 = V0
        self.phi = phi
        self.envelope_type = envelope_type
        self.mu = mu
        self.sigma = sigma
        self.t0 = t0

        # Matrici di Pauli
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    def set_pulse_rotation_numerical(self, theta, tmin, tmax, sigma_guess=1e-8):
        """
        Calcola numericamente sigma per ottenere una rotazione theta
        considerando la troncatura dell'impulso su [tmin, tmax].

        theta : angolo di rotazione desiderato (rad)
        tmin, tmax : estremi del tempo della simulazione (s)
        sigma_guess : valore iniziale per il solver (s)
        """
        if self.envelope_type != "gaussian":
            raise ValueError("Il calcolo di sigma è disponibile solo per impulsi gaussiani.")

        def integral_for_sigma(sigma):
            func = lambda t: np.exp(-0.5 * ((t - self.mu) / sigma) ** 2)
            val, _ = quad(func, tmin, tmax, limit=200)
            return V0 * val

        def objective(sigma):
            return integral_for_sigma(sigma) - theta

        sol = root_scalar(objective, bracket=[1e-12, 1e-6], method='brentq')
        if not sol.converged:
            raise RuntimeError("La ricerca di sigma non è convergente.")
        self.sigma = sol.root

        return self.sigma
    # ---------- Envelope ----------
    def envelope(self, t):
        if self.envelope_type == "gaussian":
            return np.exp(-0.5 * ((t - self.mu) / self.sigma) ** 2)
        elif self.envelope_type == "rectangular":
            return np.where((t >= self.t0) & (t <= self.t0 + self.sigma), 1.0, 0.0)
        else:
            raise ValueError("Envelope type not recognized")

    def plot_envelope(self, tlist):
        """
        Mostra l'envelope s(t) e il segnale di drive Vd(t).
        """
        s = self.envelope(tlist)
        V = self.voltage(tlist)

        fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

        ax[0].plot(tlist * 1e9, s, color='tab:blue')
        ax[0].set_ylabel("s(t)")
        ax[0].set_title(f"Envelope ({self.envelope_type})")
        ax[0].grid(True)

        ax[1].plot(tlist * 1e9, V, color='tab:orange')
        ax[1].set_xlabel("Tempo (ns)")
        ax[1].set_ylabel("Vd(t)")
        ax[1].set_title("Segnale di drive")
        ax[1].grid(True)

        plt.tight_layout()
        return fig

    # ---------- Drive Voltage ----------
    def voltage(self, t):
        s = self.envelope(t)
        return self.V0 * s * np.sin(self.wd * t + self.phi)

    # ---------- Hamiltoniane ----------
    def H0(self):
        return -0.5 * self.wq * self.sigma_z

    def Hd(self, t):
        return self.voltage(t) * self.sigma_y

    def H(self, t):
        return self.H0() + self.Hd(t)

    # ---------- Equazione di Schrödinger ----------
    def schrodinger(self, t, psi):
        return -1j * self.H(t) @ psi

    # ---------- Evoluzione temporale (Heun) ----------
    def evolve(self, psi0, tlist):
        dt = tlist[1] - tlist[0]
        psi_t = np.zeros((len(tlist), len(psi0)), dtype=complex)
        psi_t[0] = psi0

        for i in range(len(tlist) - 1):
            t = tlist[i]
            k1 = self.schrodinger(t, psi_t[i])
            k2 = self.schrodinger(t + dt, psi_t[i] + dt * k1)
            psi_t[i + 1] = psi_t[i] + (dt / 2) * (k1 + k2)
            psi_t[i + 1] /= np.linalg.norm(psi_t[i + 1])  # normalizza
        self.psi_t = psi_t
        return psi_t

    # ---------- Plot popolazioni ----------
    def plot_populations(self, tlist):
        p0 = np.abs(self.psi_t[:, 0]) ** 2
        p1 = np.abs(self.psi_t[:, 1]) ** 2

        fig = plt.figure(figsize=(7, 4))
        plt.plot(tlist * 1e9, p0, label="|0⟩")
        plt.plot(tlist * 1e9, p1, label="|1⟩")
        plt.xlabel("Tempo (ns)")
        plt.ylabel("Popolazione")
        plt.legend()
        plt.grid(True)
        return fig

    def plot_bloch_trajectory(self, tlist):
        # Matrici di Pauli
        sx, sy, sz = self.sigma_x, self.sigma_y, self.sigma_z

        # Calcola coordinate Bloch per ogni istante di tempo
        bloch = np.array([
            [np.vdot(psi, sx @ psi).real,
             np.vdot(psi, sy @ psi).real,
             np.vdot(psi, sz @ psi).real]
            for psi in self.psi_t
        ])

        # Crea figura 3D
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Traiettoria nel tempo
        ax.plot(bloch[:, 0], bloch[:, 1], bloch[:, 2], color='tab:purple', lw=2)

        # Punti di inizio (rosso) e fine (verde)
        ax.scatter(bloch[0, 0], bloch[0, 1], bloch[0, 2],
                   color='red', s=80, label='Inizio')
        ax.scatter(bloch[-1, 0], bloch[-1, 1], bloch[-1, 2],
                   color='green', s=80, label='Fine')

        # Disegna la sfera di Bloch
        u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)

        # Label e formato
        ax.set_xlabel('⟨σx⟩')
        ax.set_ylabel('⟨σy⟩')
        ax.set_zlabel('⟨σz⟩')
        ax.set_title("Traiettoria sulla Bloch Sphere")
        ax.legend()
        ax.set_box_aspect([1, 1, 1])  # sfera non distorta
        plt.tight_layout()
        return fig

# Parametri
V0 = 1e8          # rad/s
mu = 25e-9        # centro impulso
theta = np.pi/2     # X-gate
tmin, tmax = 0, 50e-9

# Tempo e stato iniziale
tlist = np.linspace(0, 50e-9, 2000)
psi0 = np.array([1, 0], dtype=complex)

# Simulazione
transmon = TransmonSimulator(wq=2*np.pi*5e8, wd=2*np.pi*5e8, V0=V0, phi=np.pi, mu=mu, envelope_type="gaussian")
sigma = transmon.set_pulse_rotation_numerical(theta, tmin, tmax)
print(f"Sigma calcolato numericamente: {sigma*1e9:.3f} ")
transmon_with_envelope = transmon.envelope(tlist)
transmon.evolve(psi0, tlist)
fig1 = transmon.plot_populations(tlist)
fig2 = transmon.plot_envelope(tlist)
fig3 = transmon.plot_bloch_trajectory(tlist)
plt.show()

