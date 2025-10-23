import numpy as np
from scipy.special import erfc


class RandomExcursionsVariantTest:
    """
    Implementation of NIST's Random Excursions Variant Test

    This test detects deviations from the expected number of visits to various states
    in the random walk. Unlike the Random Excursions test, this variant examines visits
    to a wider range of states and considers the total number of visits across all cycles.
    """

    def __init__(self, significance_level: float = 0.01):
        """
        Initialize the Random Excursions Variant Test

        Args:
            significance_level: The significance level. Default is 0.01 (1%)
        """
        self.significance_level = significance_level
        self.name = 'Random Excursions Variant Test'

        # States to examine (exactly as in NIST implementation)
        self.state_x = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def _convert_to_binary_list(self, input_data: str | bytes | list[int] | np.ndarray) -> np.ndarray:
        """Convert various input formats to a numpy array of integers (0s and 1s)"""
        if isinstance(input_data, str):
            return np.array([int(bit) for bit in input_data])
        if isinstance(input_data, bytes):
            binary_str = ''.join(format(byte, '08b') for byte in input_data)
            return np.array([int(bit) for bit in binary_str])
        if isinstance(input_data, (list, np.ndarray)):
            return np.array(input_data)
        raise ValueError('Unsupported input format')

    def test(self, binary_data: str | bytes | list[int] | np.ndarray) -> dict:
        """
        Run the Random Excursions Variant Test

        Args:
            binary_data: The sequence to test

        Returns:
            dict: Test results containing:
                - success: Boolean indicating if test was passed
                - p_values: List of p-values for each state
                - statistics: Additional test statistics
        """
        # Convert input to binary sequence
        sequence = self._convert_to_binary_list(binary_data)
        n = len(sequence)

        # Calculate the partial sums (exactly as in NIST implementation)
        S_k = np.zeros(n, dtype=int)
        S_k[0] = 2 * sequence[0] - 1
        for i in range(1, n):
            S_k[i] = S_k[i - 1] + 2 * sequence[i] - 1

        # Count cycles (returns to zero)
        J = np.sum(S_k == 0)
        if S_k[-1] != 0:  # Add one if sequence doesn't end at zero
            J += 1

        # Check if we have enough cycles
        constraint = max(0.005 * np.sqrt(n), 500)
        if constraint > J:
            return {
                'success': False,
                'error': 'Insufficient number of cycles',
                'statistics': {'n': n, 'num_cycles': J, 'constraint': constraint},
            }

        # Calculate visits and p-values for each state
        p_values = []
        visits = []

        for x in self.state_x:
            # Count visits to state x
            count = np.sum(S_k == x)

            # Calculate p-value using erfc (as in NIST implementation)
            p_value = erfc(abs(count - J) / np.sqrt(2.0 * J * (4.0 * abs(x) - 2)))

            p_values.append(p_value)
            visits.append(int(count))

        # Prepare statistics
        stats = {
            'n': n,
            'num_cycles': J,
            'states': self.state_x.tolist(),
            'visits': visits,
            'constraint': constraint,
        }

        return {
            'success': all(p >= self.significance_level for p in p_values),
            'p_values': float(p_values),
            'statistics': stats,
        }
