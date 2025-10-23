import numpy as np
from scipy.special import gammaincc
from numba import jit

class SerialTest:
    """
    Implementation of NIST's Serial Test

    This test focuses on the frequency of all possible overlapping m-bit patterns
    across the entire sequence. The purpose of this test is to determine whether
    the number of occurrences of the 2^m m-bit overlapping patterns is approximately
    the same as would be expected for a random sequence.
    """

    def __init__(self, pattern_length: int = 16, significance_level: float = 0.01):
        """
        Initialize the Serial Test

        Args:
            pattern_length: Length m of each pattern (block). Default is 16
            significance_level: The significance level. Default is 0.01 (1%)
        """
        self.pattern_length = pattern_length
        self.significance_level = significance_level
        self.name = 'Serial Test'

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

    def _psi2(self, m: int, sequence: np.ndarray) -> float:
        """
        Calculate ψ² statistic for given pattern length m

        Args:
            m: Pattern length
            sequence: Binary sequence to analyze

        Returns:
            float: ψ² statistic
        """
        seq_int32 = sequence.astype(np.int32)
        return _psi2_jit(seq_int32, m)

    def test(self, binary_data: str | bytes | list[int] | np.ndarray) -> dict:
        """
        Run the Serial Test

        Args:
            binary_data: The sequence to test

        Returns:
            dict: Test results containing:
                - success: Boolean indicating if test was passed
                - p_values: The two calculated p-values
                - statistics: Additional test statistics
        """
        # Convert input to binary sequence
        sequence = self._convert_to_binary_list(binary_data)
        n = len(sequence)
        m = self.pattern_length

        # Check if we have enough data
        if n < 2**m:
            return {
                'success': False,
                'error': f'Insufficient data. Need at least 2^{m} bits.',
                'statistics': {'n': n, 'm': m},
            }

        # Calculate ψ² statistics for m, m-1, and m-2
        psim0 = self._psi2(m, sequence)
        psim1 = self._psi2(m - 1, sequence)
        psim2 = self._psi2(m - 2, sequence)

        # Calculate the test statistics
        del1 = psim0 - psim1
        del2 = psim0 - 2.0 * psim1 + psim2

        # Calculate p-values
        p_value1 = gammaincc(2 ** (m - 1) / 2, del1 / 2.0)
        p_value2 = gammaincc(2 ** (m - 2) / 2, del2 / 2.0)

        # Prepare statistics
        stats = {
            'n': n,
            'm': m,
            'psi_m': psim0,
            'psi_m_minus_1': psim1,
            'psi_m_minus_2': psim2,
            'del1': del1,
            'del2': del2,
        }

        return {
            'success': bool(p_value1 >= self.significance_level and p_value2 >= self.significance_level),
            'p_value1': float(p_value1),
            'p_value2': float(p_value2),
            'statistics': stats,
        }


@jit(nopython=True, cache=True)
def _psi2_jit(sequence, m):
    """Calculate ψ² statistic for given pattern length m - JIT optimized"""
    if m <= 0:
        return 0.0

    n = len(sequence)
    # Initialize frequency array for all possible m-bit patterns
    pattern_counts = np.zeros(2 ** (m + 1) - 1, dtype=np.uint32)

    # Count frequencies of m-bit patterns
    for i in range(n):
        # Get m bits starting at position i (with wraparound)
        pattern = 1  # Start with 1 as in the C implementation
        for j in range(m):
            idx = (i + j) % n
            if sequence[idx] == 0:
                pattern *= 2
            else:
                pattern = 2 * pattern + 1
        pattern_counts[pattern - 1] += 1

    # Calculate ψ² statistic
    # Only use counts for complete m-bit patterns (last 2^m entries)
    start_idx = 2 ** m - 1
    end_idx = 2 ** (m + 1) - 1
    relevant_counts = pattern_counts[start_idx:end_idx]

    sum_squares = 0.0
    for count in relevant_counts:
        sum_squares += count * count

    return (2 ** m / n) * sum_squares - n