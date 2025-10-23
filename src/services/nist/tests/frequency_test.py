import numpy as np
from scipy.stats import norm
import math


class FrequencyTest:
    """
    Implementation of NIST's Frequency (Monobit) Test

    This test focuses on the proportion of zeroes and ones for the entire sequence.
    The purpose of this test is to determine whether the number of ones and zeros in
    a sequence are approximately the same as would be expected for a truly random sequence.
    """

    def __init__(self, significance_level: float = 0.01):
        """
        Initialize the Frequency Test

        Args:
            significance_level: The significance level. Default is 0.01 (1%)
        """
        self.significance_level = significance_level
        self.name = 'Frequency (Monobit) Test'

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
            # Convert each byte to its binary representation
            binary_str = ''.join(format(byte, '08b') for byte in input_data)
            return np.array([int(bit) for bit in binary_str])
        if isinstance(input_data, (list, np.ndarray)):
            return np.array(input_data)
        raise ValueError('Unsupported input format')

    def test(self, binary_data: str | bytes | list[int] | np.ndarray) -> dict:
        """
        Run the Frequency (Monobit) Test

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

        # Convert zeros to -1, ones stay as 1
        s = 2 * sequence - 1

        # Calculate test statistics
        s_obs = abs(s.sum()) / math.sqrt(n)
        p_value = norm.sf(s_obs / math.sqrt(2))

        # Prepare statistics
        stats = {
            'n': n,
            'ones_count': np.sum(sequence == 1),
            'zeros_count': np.sum(sequence == 0),
            'ones_proportion': np.mean(sequence),
            'partial_sum': s.sum(),
            's_obs': s_obs,
        }

        return {
            'success': bool(p_value >= self.significance_level),
            'p_value': float(p_value),
            'statistics': stats,
        }
