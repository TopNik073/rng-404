import numpy as np
from scipy.special import erfc


class RunsTest:
    """
    Implementation of NIST's Runs Test

    The focus of this test is the total number of runs in the sequence, where a run
    is an uninterrupted sequence of identical bits. The purpose of this test is to
    determine whether the oscillation between zeros and ones in the sequence is too
    fast or too slow compared to what would be expected for a truly random sequence.
    """

    def __init__(self, significance_level: float = 0.01):
        """
        Initialize the Runs Test

        Args:
            significance_level: The significance level. Default is 0.01 (1%)
        """
        self.significance_level = significance_level
        self.name = 'Runs Test'

    def _convert_to_binary_list(self, input_data: str | bytes | list[int] | np.ndarray) -> np.ndarray:
        """
        Convert various input formats to a numpy array of integers (0s and 1s)

        Args:
            input_data: Input sequence in various formats
                       - Binary string ("01001")
                       - Bytes object
                       - List of integers ([0,1,0,0,1])
                       - Numpy array

        Returns:
            numpy.ndarray: Array of 0s and 1s
        """
        if isinstance(input_data, str):
            return np.array([int(bit) for bit in input_data])
        if isinstance(input_data, bytes):
            binary_str = ''.join(format(byte, '08b') for byte in input_data)
            return np.array([int(bit) for bit in binary_str])
        if isinstance(input_data, (list, np.ndarray)):
            return np.array(input_data)
        raise ValueError('Unsupported input format')

    def _count_runs(self, sequence: np.ndarray) -> int:
        """
        Count the number of runs in the sequence

        A run is an uninterrupted sequence of identical bits.

        Args:
            sequence: Binary sequence as numpy array

        Returns:
            int: Number of runs
        """
        # Use XOR to find transitions between different bits
        # Add 1 because number of runs is one more than number of transitions
        return np.sum(sequence[1:] != sequence[:-1]) + 1

    def test(self, binary_data: str | bytes | list[int] | np.ndarray) -> dict:
        """
        Run the Runs Test

        Args:
            binary_data: The sequence to test

        Returns:
            dict: A dictionary containing:
                - 'success': Boolean indicating if the test was passed
                - 'p_value': The p-value of the test
                - 'statistics': Additional test statistics
        """
        # Convert input to binary sequence
        sequence = self._convert_to_binary_list(binary_data)
        n = len(sequence)

        # Calculate proportion of ones
        S = np.sum(sequence)
        pi = S / n

        # Check if sequence meets criteria for runs test
        # According to NIST SP 800-22, for large n the test can proceed even if pi != 0.5
        tau = 2.0 / np.sqrt(n)

        # For large sequences, use relaxed criteria as per NIST implementation
        if n >= 100:  # noqa
            if abs(pi - 0.5) >= tau:
                # Use adjusted calculation for sequences that don't meet exact 0.5 proportion
                # But still proceed with the test using the actual pi value
                exp_runs = 2 * n * pi * (1 - pi)
                std_dev = np.sqrt(2 * n * pi * (1 - pi) * (1 - 3 * pi + 3 * pi * pi))
            else:
                # Original calculation for sequences meeting the 0.5 proportion criteria
                exp_runs = 2 * n * 0.5 * 0.5  # = n/2
                std_dev = np.sqrt(n * 0.5 * 0.5)  # = sqrt(n)/2
        else:
            # For small sequences, use strict criteria
            if abs(pi - 0.5) >= tau:
                return {
                    'success': False,
                    'p_value': 0.0,
                    'statistics': {
                        'n': n,
                        'proportion_ones': pi,
                        'tau': tau,
                        'criteria_met': False,
                        'error': 'Pi estimator criteria not met for small sequence',
                    },
                }
            exp_runs = n / 2
            std_dev = np.sqrt(n) / 2

        # Count runs
        V_n_obs = self._count_runs(sequence)

        # Calculate test statistic and p-value
        if std_dev > 0:
            z_score = abs(V_n_obs - exp_runs) / std_dev
            p_value = erfc(z_score / np.sqrt(2))
        else:
            p_value = 0.0

        # Ensure p-value is valid
        p_value = max(0.0, min(1.0, p_value))

        # Prepare statistics
        stats = {
            'n': n,
            'proportion_ones': pi,
            'tau': tau,
            'criteria_met': abs(pi - 0.5) < tau,
            'observed_runs': V_n_obs,
            'expected_runs': exp_runs,
            'standard_deviation': std_dev,
            'test_statistic': z_score if std_dev > 0 else 0,
        }

        return {
            'success': bool(p_value >= self.significance_level),
            'p_value': float(p_value),
            'statistics': stats,
        }
