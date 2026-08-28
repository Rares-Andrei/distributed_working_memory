from difflib import get_close_matches

class WorkingMemoryParameters:
    """
    This class includes various parameters involved in the dynamics of a working memory model.
    Parameters are stored in a dictionary for flexibility and can be updated using the `update` method.
    """

    def __init__(self):
        # Directly initializing self._params
        self.__dict__["_params"] = {
            # I -> r
            "A": 135,           # Hz/nA
            "B": 54,            # Hz
            "D": 0.308,         # s
            "G_I": 4,
            "C_0": 177,         # Hz
            "C_1": 615,         # Hz/nA
            "R_0": 5.5,         # Hz
            "TAU_R": 2e-3,      # s

            # r -> S
            "TAU_N": 60e-3,     # s
            "TAU_G": 5e-3,      # s
            "GAMMA": 1.282,
            "GAMMA_I": 2.0,

            # S -> I
            "J_S": 0.3213,      # nA
            "J_C": 0.0107,      # nA
            "J_IE": 0.15,       # nA
            "J_EI": -0.31,      # nA
            "J_II": -0.12,      # nA
            "I_0A": 0.3294,     # nA
            "I_0B": 0.3294,     # nA
            "I_0C": 0.26,       # nA

            # Noise
            "TAU_NOISE": 2e-3,  # s
            "SIGMA_A": 0.01,    # nA
            "SIGMA_B": 0.01,    # nA

            # Gradient of J_s
            "J_MIN": 0.21,      # nA
            "J_MAX": 0.42,      # nA (< 0.4655)

            # Gradient of J_IE
            "J_0": 0.2112,      # nA
            "ZETA": None,

            # Inter-areal projections
            "G": 0.48,          # global coupling strength
            "Z": None,          # E-I balancing factor
            "ALPHA": 1          # relative strength of (feedback) inhibitory projections
        }

        # Initialize dependent parameters
        self._recalculate_dependent_parameters()

    def __getattr__(self, name):
        """
        Dynamically retrieve the value of a parameter.
        """
        # Safely access self._params using __dict__ to avoid recursion
        params = self.__dict__.get("_params", {})
        if name in params:
            return params[name]
        
        # Suggest similar parameter names
        similar_keys = get_close_matches(name, params.keys(), n=3, cutoff=0.5)
        if similar_keys:
            suggestion = f" Did you mean one of these? {similar_keys}"
        else:
            suggestion = f" Available parameters are: {list(params.keys())}"
        
        raise AttributeError(
            f"'WorkingMemoryParameters' object has no attribute '{name}'."
            f"{suggestion}"
        )

    def update(self, **kwargs):
        """
        Update parameters of the model.
        """
        dependent_params = ["ZETA", "Z"]
        for key, value in kwargs.items():
            if key in dependent_params:
                raise AttributeError(f"Cannot directly update dependent parameter '{key}'.")
            if key.upper() in map(str.upper, self._params.keys()):  # Ignore case
                matching_key = [k for k in self._params.keys() if k.upper() == key.upper()][0]
                self._params[matching_key] = value
            else:
                raise AttributeError(f"Invalid parameter '{key}'. Available parameters are: {list(self._params.keys())}")
        self._recalculate_dependent_parameters()

    def _recalculate_dependent_parameters(self):
        """
        Recalculate dependent parameters after updating.
        """
        self._params["ZETA"] = self.calc_zeta()
        self._params["Z"] = self.calc_Z()

    def calc_zeta(self):
        """
        Compute the inhibitory-to-excitatory balancing factor (ZETA).
        """
        tau_g = self._params["TAU_G"]
        gamma_i = self._params["GAMMA_I"]
        c_1 = self._params["C_1"]
        j_ii = self._params["J_II"]
        g_i = self._params["G_I"]

        denominator = g_i - j_ii * tau_g * gamma_i * c_1
        if denominator <= 0:
            raise ValueError("Denominator in ZETA calculation is non-positive.")
        return (tau_g * gamma_i * c_1) / denominator

    def calc_Z(self):
        """
        Compute the E-I balancing factor (Z).
        """
        tau_g = self._params["TAU_G"]
        gamma_i = self._params["GAMMA_I"]
        c_1 = self._params["C_1"]
        j_ei = self._params["J_EI"]
        j_ii = self._params["J_II"]
        g_i = self._params["G_I"]

        denominator = c_1 * tau_g * gamma_i * j_ii - g_i
        if denominator == 0:
            raise ValueError("Denominator in Z calculation is zero.")
        return (2 * c_1 * tau_g * gamma_i * j_ei) / denominator
