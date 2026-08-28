import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix
plot_params = {
    "figure.dpi": "600",
    "axes.labelsize": 20,
    "axes.linewidth": 1.5,
    "axes.titlesize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.title_fontsize": 16,
    "legend.fontsize": 14,
    "xtick.major.size": 3.5,
    "xtick.major.width": 1.5,
    "xtick.minor.size": 2.5,
    "xtick.minor.width": 1.5,
    "ytick.major.size": 3.5,
    "ytick.major.width": 1.5,
    "ytick.minor.size": 2.5,
    "ytick.minor.width": 1.5,
}
plt.rcParams.update(plot_params)
class WorkingMemoryNetwork:
    def __init__(self, area_names, area_lobes, Y0, W, F, h, params, remove_FB=False, random_seed=None):
        """
        Initialize the network object for the working memory model.

        Parameters
        ----------
        area_names: list
            List of brain area names.
        area_lobes: list
            List of brain area lobes.
        Y0: np.ndarray
            N x 8 matrix of initial conditions. Each row is a brain area, each column is a variable vector (r_A, r_B, r_C, S_A, S_B, S_C, x_A, x_B).
        W: np.ndarray
            N x N matrix of connectivity strengths.
        F: np.ndarray
            N x N matrix of feedforward (FF) relative strengths.
        h: np.ndarray
            1D array of the positions of the brain areas in the T1w:T2w ranking.
        params: object
            Parameters object containing the parameters of the model.
        remove_FB: bool, optional
            Whether to remove feedback projections. Default is False.
        random_seed: int, optional
            Random seed for reproducibility. Default is None.
        """
        self.area_names = area_names.copy()
        self.area_lobes = area_lobes.copy()
        self.N = len(area_names)    # number of brain areas

        # Check the dimensions of the input matrices
        assert len(area_lobes) == self.N, f"Length of area_lobes must be {self.N}!"
        assert Y0.shape == (self.N, 8), f"Y0 must be a {self.N} x 8 matrix!"
        assert W.shape == (self.N, self.N), f"W must be a {self.N} x {self.N} matrix!"
        assert F.shape == (self.N, self.N), f"F must be a {self.N} x {self.N} matrix!"
        assert self.N == h.shape[0], f"The length of h must be {self.N}!"
        assert np.min(W) >= 0, "Ensure all connections are non-negative!"
        assert np.all(np.diag(W) == 0), "Ensure no self-connections!"
        assert np.min(F) >= 0 and np.max(F) <= 1, "Ensure all feedforward relative strengths are between 0 and 1!"
        assert np.min(h) >= 0 and np.max(h) <= 1, "Ensure all hierarchical level values are between 0 and 1!"

        self.Y0 = Y0.copy()     # store initial conditions
        self.Y = Y0.copy()      # initialize current state
        self.W = W.copy()       # connectivity matrix (FLN)
        self.F = F.copy()       # FF relative strength matrix (SLN)
        self.h = h.copy()       # hierarchical level
        self.params = params

        self.J_s = self.calc_J_s()      # calculate gradient of J_s
        self.J_IE = self.calc_J_IE()    # calculate gradient of J_IE

        # "SLN-driven modulation of FB projections between frontal areas is not too large, so that interactions between these areas are never strongly inhibitory. In practice, such constraint is only necessary for projections from frontal areas to 8 l and 8 m. We consider that the SLN-driven modulation of FB projections to 8 l and 8 m is never larger than 0.4."
        self.lambda_I = 1 - self.F   # relative strength of (feedback) inhibitory projections
        self.frontal_indices = [i for i in range(self.N) if self.area_lobes[i] in ('Frontal', 'Prefrontal')]
        inhib_max = 0.4     # maximum inhibitory strength
        for i in range(2):  # 8l and 8m are the first two frontal areas
            for source_area_idx in self.frontal_indices:
                x_idx = self.frontal_indices[i]
                y_idx = source_area_idx
                self.lambda_I[x_idx, y_idx] = min(self.lambda_I[x_idx, y_idx], inhib_max)

        # Precompute G * W * F and G/Z * W * (1-F) with gradients of long-range projection strengths
        self.W_E = self.params.G * self.W * np.outer(self.J_s / np.max(self.J_s), np.ones(self.N)) * self.F
        self.W_I = self.params.ALPHA * self.params.G / self.params.Z * self.W * np.outer(self.J_IE / np.max(self.J_IE), np.ones(self.N)) * self.lambda_I
        if remove_FB:   # remove feedback projections
            self.W_E -= np.triu(self.W_E, k=1)
            self.W_I -= np.triu(self.W_I, k=1)
        
        self.state_histories = []   # list to store the state history
        self.time_histories = []    # list to store the time history
        self.monitor = []   # list to store the monitored variables

        if random_seed:
            self.random_seed = random_seed
            self.rng = np.random.default_rng(random_seed)
        else:
            self.random_seed = None
            self.rng = np.random.default_rng()

    def calc_J_s(self):
        """
        Calculate the synaptic strength J_s for each brain area.
        """
        return self.params.J_MIN + (self.params.J_MAX - self.params.J_MIN) * self.h

    def calc_J_IE(self):
        """
        Calculate the synaptic strength J_IE for each brain area.
        """
        return (self.params.J_0 - self.J_s - self.params.J_C) / (2 * self.params.J_EI * self.params.ZETA)

    def phi_E(self, I):
        """
        Transfer function for excitatory populations.
        """
        return (self.params.A * I - self.params.B) / (1 - np.exp(-self.params.D * (self.params.A * I - self.params.B)))

    def phi_I(self, I):
        """
        Transfer function for inhibitory populations.
        """
        return np.maximum(self.params.R_0 + (self.params.C_1 * I - self.params.C_0) / self.params.G_I, 0)

    def calc_dY(self, Y, dt, dt_over_taus, I_ext):
        """
        Calculate the differential of Y for this N x 8 system.

        Parameters
        ----------
        Y: np.ndarray
            N x 8 matrix of variables (r_A, r_B, r_C, S_A, S_B, S_C, x_A, x_B).
        dt: float
            Time step size.
        dt_over_taus: np.ndarray
            Precomputed array of time step ratios (dt/tau_r, dt/tau_N, dt/tau_G, dt/tau_noise, sqrt(dt/tau_noise)).
        I_ext: np.ndarray
            N x 3 matrix of external inputs. Each row is a brain area, each column is an external input (I_ext_A, I_ext_B, I_ext_C).

        Returns
        -------
        dY: np.ndarray
            N x 8 matrix of differentials.
        """
        # Extract variables and precomputed time step ratios
        r_A, r_B, r_C = Y[:, 0], Y[:, 1], Y[:, 2]
        S_A, S_B, S_C = Y[:, 3], Y[:, 4], Y[:, 5]
        x_A, x_B = Y[:, 6], Y[:, 7]
        dt_over_tau_r, dt_over_tau_N, dt_over_tau_G, dt_over_tau_noise, sqrt_dt_over_tau_noise = dt_over_taus

        # Compute the inputs
        I_A = self.J_s * S_A + self.params.J_C * S_B + self.params.J_EI * S_C + self.params.I_0A + self.W_E @ S_A + x_A + I_ext[:, 0]
        I_B = self.params.J_C * S_A + self.J_s * S_B + self.params.J_EI * S_C + self.params.I_0B + self.W_E @ S_B + x_B + I_ext[:, 1]
        I_C = self.J_IE * S_A + self.J_IE * S_B + self.params.J_II * S_C + self.params.I_0C + self.W_I @ (S_A + S_B) + I_ext[:, 2]

        # Compute the differentials of firing rates
        dr_A = (-r_A + self.phi_E(I_A)) * dt_over_tau_r
        dr_B = (-r_B + self.phi_E(I_B)) * dt_over_tau_r
        dr_C = (-r_C + self.phi_I(I_C)) * dt_over_tau_r

        # Compute the differentials of synaptic conductances
        dS_A = -S_A * dt_over_tau_N + self.params.GAMMA * (1 - S_A) * r_A * dt
        dS_B = -S_B * dt_over_tau_N + self.params.GAMMA * (1 - S_B) * r_B * dt
        dS_C = -S_C * dt_over_tau_G + self.params.GAMMA_I * r_C * dt

        # Compute the differentials of noise terms (without instantaneous part)
        dx_A = -x_A * dt_over_tau_noise + self.params.SIGMA_A * sqrt_dt_over_tau_noise * self.rng.normal(loc=0, scale=1, size=(self.N,))
        dx_B = -x_B * dt_over_tau_noise + self.params.SIGMA_B * sqrt_dt_over_tau_noise * self.rng.normal(loc=0, scale=1, size=(self.N,))

        # Combine the differentials
        dY = np.zeros_like(Y)
        dY[:, 0] = dr_A
        dY[:, 1] = dr_B
        dY[:, 2] = dr_C
        dY[:, 3] = dS_A
        dY[:, 4] = dS_B
        dY[:, 5] = dS_C
        dY[:, 6] = dx_A
        dY[:, 7] = dx_B

        return dY

    def precompute_dt_over_taus(self, dt):
        """
        Precompute time step ratios to avoid recalculating during iterations.

        Parameters
        ----------
        dt: float
            Time step size.

        Returns
        -------
        dt_over_taus: np.ndarray
            Array of time step ratios and their square root for noise calculations.
        """
        taus = np.array([self.params.TAU_R, self.params.TAU_N, self.params.TAU_G, self.params.TAU_NOISE])
        remainders = taus % dt
        assert remainders.all() == 0, f"dt is suggested to be divisible by all time constants: {taus}"
        dt_over_taus = dt / taus
        return np.append(dt_over_taus, np.sqrt(dt_over_taus[-1]))  # Add sqrt(dt/tau_noise) for noise

    def run(self, t_end, dt, t_start=0, I_ext_strengths=None, ts_ext_start=None, ts_ext_end=None, lesion_areas=None):
        """
        Run the simulation.

        Parameters
        ----------
        t_end: float
            End time of the simulation.
        dt: float
            Time step.
        t_start: float, optional
            Start time of the simulation. Default is 0.
        I_ext_strengths: np.ndarray, optional
            N_period x N x 3 matrix of external inputs strengths. N_period is the number of input periods. In each input period, the external input strength is a N x 3 matrix. If None, no external input.
        ts_ext_start: np.ndarray, optional
            Start times of the external input. If None, no external input.
        ts_ext_end: np.ndarray, optional
            End times of the external input. If None, no external input.
        lesion_areas: list, optional
            List of brain areas to lesion. If None, no lesion.

        Returns
        -------
        state_history: np.ndarray
            State history in this simulation (len(t_eval) x N x 8 array).
        """
        # Generate time points
        t_eval = np.arange(t_start, t_end+dt, dt)
        self.time_histories.append(t_eval)

        # Check if external input is provided
        if I_ext_strengths is None:
            ts_ext_start, ts_ext_end = None, None
        else:
            assert I_ext_strengths.shape[0] == len(ts_ext_start) == len(ts_ext_end), "Number of external input periods should be consistent!"
            assert I_ext_strengths.shape[1:] == (self.N, 3), "In each external input period, I_ext_strength should be an N x 3 matrix!"
            for t_ext_start, t_ext_end in zip(ts_ext_start, ts_ext_end):
                assert t_ext_start < t_ext_end, "ts_ext_start must be less than ts_ext_end!"
                assert t_ext_start >= t_start and t_ext_end <= t_end, "External input time must be within the simulation time!"

        # Precompute time step ratios
        dt_over_taus = self.precompute_dt_over_taus(dt)

        # Euler-Maruyama integration
        Y = self.Y.copy()
        state_history = [Y.copy()]
        for i, t in enumerate(t_eval[1:], start=1):
            # Generate I_ext
            I_ext = np.zeros((self.N, 3))
            if ts_ext_start is not None:    # if external input is provided
                for j, (t_ext_start, t_ext_end) in enumerate(zip(ts_ext_start, ts_ext_end)):    # loop over external input periods
                    if t_ext_start <= t < t_ext_end:
                        I_ext += I_ext_strengths[j]

            # Calculate the differentials
            dY = self.calc_dY(Y, dt, dt_over_taus, I_ext)

            # Update the state
            Y += dY

            # Apply lesion if necessary
            if lesion_areas:
                for lesion_area in lesion_areas:
                    area_idx = np.where(np.array(self.area_names) == lesion_area)[0][0]
                    Y[area_idx, :6] = 0

            # Store the current state
            state_history.append(Y.copy())

        # Store the state history
        self.state_histories.append(np.array(state_history))

        # Update the current state
        self.Y = Y

        return np.array(state_history)

    def get_state_histories(self, variables=['all']):
        """
        Get the state histories of the network.

        Parameters
        ----------
        variables: list
            list of variables to return. If ['all'], return all variables.
        Returns
        -------
        state_histories: list
            List of state matrices for all simulations (length N_sim). Each element is a N_t x N x N_var matrix.
        """
        if 'all' in variables:
            return self.state_histories

        hash_map = {'r_A': 0, 'r_B': 1, 'r_C': 2, 'S_A': 3, 'S_B': 4, 'S_C': 5, 'x_A': 6, 'x_B': 7}
        indices = [hash_map[variable] for variable in variables]

        extracted_histories = []
        for state_history in self.state_histories:
            # state_matrix is of shape (time_steps, N, 8)
            # Extract the desired variables along the last axis
            extracted_history = state_history[:, :, indices]
            extracted_histories.append(extracted_history)

        return extracted_histories

    def merge_histories(self):
        """
        Merge the time histories and state histories of all simulations respectively.

        Returns
        -------
        merged_time: np.ndarray
            Merged time histories (length N_t_total).
        merged_state: np.ndarray
            Merged state histories (N_t_total x N x 8).
        """
        # Check if history is available
        if not self.time_histories or not self.state_histories:
            raise ValueError("No history to merge. Run a simulation first.")

        # Initialize merged arrays
        merged_time = self.time_histories[0]
        merged_state = self.state_histories[0]

        # Iterate over the remaining histories
        for i in range(1, len(self.time_histories)):
            time = self.time_histories[i]
            state = self.state_histories[i]

            # Concatenate time and state
            merged_time = np.concatenate((merged_time, time[1:] + merged_time[-1] - time[0]))
            merged_state = np.concatenate((merged_state, state[1:]), axis=0)

        return merged_time, merged_state

    def clear_histories(self):
        """
        Clear the time histories and state histories.
        """
        self.state_histories = []
        self.time_histories = []
        self.monitor = []

    def reset(self, Y0=None, random_seed=None):
        """
        Reset the network to a given initial state or the original initial state.

        Parameters
        ----------
        Y0: np.ndarray, optional
            N x 8 matrix of initial conditions. Each row is a brain area, each column is a variable vector (r_A, r_B, r_C, S_A, S_B, S_C, x_A, x_B). If None, use the original initial state.
        random_seed: int, optional
            Random seed for reproducibility. If None, use the original random seed.
        """
        self.Y0 = Y0 or self.Y0     # if Y0 is not provided, use the original initial state
        self.Y = self.Y0    # reset the state
        self.clear_histories()
        if random_seed:
            self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)  # reset the random number generator

    def plot_all_areas(self, xlim, axes_flat=None, variables=['r_A', 'r_B'], colors=[[.1, .6, .8], [.6, 0, .5]], title=None, save_path=None, downsample=10, legend=True):
        """
        Plot the variables of all brain areas.

        Parameters
        ----------
        xlim: tuple
            Tuple of x-axis limits (start, end).
        axes_flat: matplotlib.axes.Axes, optional
            Flattened axes object to plot on. If None, create a new figure.
        variables: list of str
            List of variables to plot. Must match variable names in state history.
        colors: list
            List of colors for each variable, must match the length of `variables`.
        title: str, optional
            Title of the plot.
        save_path: str, optional
            Path to save the plot. If None, do not save the plot.
        downsample: int, optional
            Downsampling factor for time series data to speed up plotting. Default is 10.
        legend: bool, optional
            Whether to show the legend. Default is True.

        Returns
        -------
        lines: list
            List of Line2D objects representing the plotted data.
        """
        # Check variables and colors
        if not isinstance(variables, list) or not all(isinstance(v, str) for v in variables):
            raise ValueError("`variables` must be a list of strings representing variable names.")
        assert len(variables) == len(colors), "Length of variables and colors must be the same."

        # Prepare data and area names
        state_history = self.get_state_histories(variables)[-1]
        t = self.time_histories[-1]  # Time array
        t_downsampled = t[::downsample]  # Downsample time
        t_end = t[-1]  # Use the last time point as t_end

        # Prepare plotting
        rows = math.ceil(self.N / 6)  # Determine grid rows
        existing_plots = False  # Track if there are existing plots in the axes
        if axes_flat is None:
            fig, axes = plt.subplots(rows, 6, figsize=(15, 2 * rows), dpi=100)
            fig.subplots_adjust(hspace=0.5, wspace=0.3)
            axes = axes.flatten()  # Flatten for easy iteration
        else:
            axes = axes_flat
            if axes[0].lines:
                existing_plots = True

        # Plot firing rates for each brain area
        lines = []
        for i, ax in enumerate(axes[:self.N]):  # Only iterate over valid brain areas
            ymax = 0  # Track max y value for this subplot
            for var_idx, var in enumerate(variables):
                data = state_history[:, i, var_idx][::downsample]  # Downsample data
                if i == 0:
                    line = ax.plot(t_downsampled, data, label='$'+var+'$', color=colors[var_idx], linewidth=2)
                    lines.append(line[0])
                    if legend:
                        ax.legend(loc="upper right")
                else:
                    ax.plot(t_downsampled, data, color=colors[var_idx], linewidth=2)
                ymax = max(ymax, data.max())  # Update y max value

            # Add titles, labels, and ylimits
            ax.set_title(self.area_names[i], fontweight='bold')
            if existing_plots:  # If there are existing plots before calling this function
                x_min1, x_max1 = ax.get_xlim()
                x_min2, x_max2 = xlim
                ax.set_xlim(min(x_min1, x_min2), max(x_max1, x_max2))
                y_min1, y_max1 = ax.get_ylim()
                y_min2, y_max2 = -max(ymax * 0.2, 0.5), max(ymax * 1.3, 2)
                ax.set_ylim(min(y_min1, y_min2), max(y_max1, y_max2))
            else:  # If there are no existing plots before calling this function
                ax.set_xlim(*xlim)
                ax.set_ylim(-max(ymax * 0.2, 0.5), max(ymax * 1.3, 2))

            if i % 6 == 0:  # First column: add ylabel
                ax.set_ylabel("Rate (Hz)")
            if i >= (rows - 1) * 6:  # Last row: add xlabel
                ax.set_xlabel("Time (s)")
            
            ax.grid(linestyle=':', alpha=0.5)
            for spine in ['top','right']:
                ax.spines[spine].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        # Remove empty subplots if self.N is not a multiple of 6
        for j in range(self.N, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout and add optional title
        if title:
            plt.suptitle(title, fontweight='bold')
        plt.tight_layout()

        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=1200)  # Set proper DPI for high-quality saving

        if axes_flat is None:
            plt.show()

        return lines

    def plot_n_areas(self, xlim, target_area_names=['V1', 'MT', 'LIP', '24c', 'STPi', '9/46d'], axes_flat=None, variables=['r_A', 'r_B'], colors=[[.1, .6, .8], [.6, 0, .5]], title=None, save_path=None, downsample=10, legend=True):
        """
        Plot the firing rates of selected brain areas.

        Parameters
        ----------
        xlim: tuple
            Tuple of x-axis limits (start, end).
        target_area_names: list, optional
            List of brain area names to plot.
        axes_flat: matplotlib.axes.Axes, optional
            Flattened axes object to plot on. If None, create a new figure.
        variables: list of str
            List of variables to plot. Must match variable names in state history.
        colors: list
            List of colors for each variable, must match the length of `variables`.
        title: str, optional
            Title of the plot.
        save_path: str, optional
            Path to save the plot. If None, do not save the plot.
        downsample: int, optional
            Downsampling factor for time series data to speed up plotting. Default is 10.
        legend: bool, optional
            Whether to show the legend. Default is True.

        Returns
        -------
        lines: list
            List of Line2D objects representing the plotted data.
        """
        # Check variables and colors
        if not isinstance(variables, list) or not all(isinstance(v, str) for v in variables):
            raise ValueError("`variables` must be a list of strings representing variable names.")
        assert len(variables) == len(colors), "Length of variables and colors must be the same."

        # Prepare data and area names
        N_area = len(target_area_names)
        state_history = self.get_state_histories(variables)[-1]
        t = self.time_histories[-1]  # Time array
        t_downsampled = t[::downsample]  # Downsample time
        t_end = t[-1]  # Use the last time point as t_end

        # Prepare plotting
        if N_area > 6:
            columns = 6
            rows = math.ceil(N_area / 6)
            figsize = (15, 2 * rows)
        else:
            columns = 3
            rows = math.ceil(N_area / 3)
            figsize = (8, 2.5 * rows)

        existing_plots = False  # Track if there are existing plots in the axes
        if axes_flat is None:
            fig, axes = plt.subplots(rows, columns, figsize=figsize, dpi=100)
            fig.subplots_adjust(hspace=0.5, wspace=0.3)
            axes = axes.flatten()  # Flatten for easy iteration
        else:
            axes = axes_flat
            if axes[0].lines:
                existing_plots = True

        # Plot firing rates for each brain area
        lines = []
        for i, ax in enumerate(axes[:N_area]):  # Only iterate over valid brain areas
            area_idx = np.where(np.array(self.area_names) == target_area_names[i])[0][0]
            ymax = 0  # Track max y value for this subplot
            for var_idx, var in enumerate(variables):
                data = state_history[:, area_idx, var_idx][::downsample]  # Downsample data
                if i == 0:
                    line = ax.plot(t_downsampled, data, label='$'+var+'$', color=colors[var_idx], linewidth=2)
                    lines.append(line[0])
                    if legend:
                        ax.legend(loc="upper right", fontsize=10)
                else:
                    ax.plot(t_downsampled, data, color=colors[var_idx], linewidth=2)
                ymax = max(ymax, data.max())  # Update y max value

            # Add titles, labels, and ylimits
            ax.set_title(target_area_names[i], fontsize=13, fontweight='bold')
            if existing_plots:  # If there are existing plots before calling this function
                x_min1, x_max1 = ax.get_xlim()
                x_min2, x_max2 = xlim
                ax.set_xlim(min(x_min1, x_min2), max(x_max1, x_max2))
                y_min1, y_max1 = ax.get_ylim()
                y_min2, y_max2 = -max(ymax * 0.2, 0.5), max(ymax * 1.3, 2)
                ax.set_ylim(min(y_min1, y_min2), max(y_max1, y_max2))
            else:  # If there are no existing plots before calling this function
                ax.set_xlim(*xlim)
                ax.set_ylim(-max(ymax * 0.2, 0.5), max(ymax * 1.3, 2))

            if i % columns == 0:  # First column: add ylabel
                ax.set_ylabel("Rate (sp/s)", fontsize=12)
            if i >= (rows - 1) * columns:  # Last row: add xlabel
                ax.set_xlabel("Time (s)", fontsize=12)

        # Remove empty subplots if self.N is not a multiple of 6
        for j in range(self.N, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout and add optional title
        if title:
            plt.suptitle(title, fontsize=18, fontweight='bold')
        plt.tight_layout()

        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)  # Set proper DPI for high-quality saving

        if axes_flat is None:
            plt.show()

        return lines

    def damage_local_circuits(self, cs, damage_func):
        """
        Damage local circuits in the network by modifying the synaptic strengths based on a damage function.

        Parameters
        ----------
        cs: np.ndarray
            The concentration of toxic tau proteins in each brain area.
        damage_func: function
            The damage function to apply to the synaptic strengths J_s and J_IE.
        """
        assert cs.shape == (self.N,), "Concentration of toxic tau proteins must be a 1D array of length N."

        # Apply the damage function to synaptic strengths
        eta = damage_func(cs)
        self.J_s *= eta
        self.J_IE *= eta
        
        assert eta.shape == (self.N,), "Damage function must return a 1D array of length N."

        # Recompute the synaptic strengths
        self.W_E = self.params.G * self.W * np.outer(self.J_s / np.max(self.J_s), np.ones(self.N)) * self.F
        self.W_I = self.params.G / self.params.Z * self.W * np.outer(self.J_IE / np.max(self.J_IE), np.ones(self.N)) * self.lambda_I