import numpy as np
from scipy.special import gammaincc
from numba import jit

class ApproximateEntropyTest:
    """
    Implementation of NIST's Approximate Entropy Test

    This test compares the frequency of overlapping blocks of two consecutive/adjacent
    lengths (m and m+1) against the expected result for a random sequence. The test
    assesses the uniformity and randomness of patterns in the sequence by measuring
    the logarithmic likelihood ratio of finding repeating patterns.
    """

    def __init__(self, block_length: int = 10, significance_level: float = 0.01):
        """
        Initialize the Approximate Entropy Test

        Args:
            block_length: Length m of each block. Default is 10
            significance_level: The significance level (a). Default is 0.01 (1%) # noqa
        """
        self.block_length = block_length
        self.significance_level = significance_level
        self.name = 'Approximate Entropy Test'

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

    def _compute_frequency(self, sequence: np.ndarray, block_size: int) -> tuple[np.ndarray, int]:
        """
        Compute frequencies of all possible patterns of given block size

        Args:
            sequence: Input binary sequence
            block_size: Size of blocks to analyze

        Returns:
            tuple: (frequencies, power_len)
                frequencies: Array of pattern frequencies
                power_len: Length of pattern array (2^block_size)
        """
        seq_int32 = sequence.astype(np.int32)
        return _compute_frequency_jit(seq_int32, block_size)

    def _compute_entropy(self, frequencies: np.ndarray, seq_length: int, block_size: int) -> float:
        """
        Compute entropy value for given frequencies

        Args:
            frequencies: Array of pattern frequencies
            seq_length: Length of the sequence
            block_size: Size of blocks used

        Returns:
            float: Computed entropy value
        """
        return _compute_entropy_jit(frequencies, seq_length, block_size)

    def test(self, binary_data: str | bytes | list[int] | np.ndarray) -> dict:
        """
        Run the Approximate Entropy Test

        Args:
            binary_data: The sequence to test

        Returns:
            dict: Test results containing:
                - success: Boolean indicating if test was passed
                - p_value: The calculated p-value
                - statistics: Additional test statistics
        """
        # Convert input to binary sequence
        sequence = self._convert_to_binary_list(binary_data)
        n = len(sequence)
        m = self.block_length

        # Check if we have enough data and valid block size
        if m > int(np.log2(n) - 5):
            return {
                'success': False,
                'error': (
                    f'Block size {m} exceeds recommended value of '
                    f'{max(1, int(np.log2(n) - 5))}. Results may be inaccurate.'
                ),
                'statistics': {'n': n, 'm': m},
            }

        # Calculate ApEn(m) and ApEn(m+1)
        ApEn = np.zeros(2)

        for block_size in [m, m + 1]:
            if block_size == 0:
                ApEn[0] = 0.0
            else:
                # Compute pattern frequencies
                frequencies, _power_len = self._compute_frequency(sequence, block_size)

                # Compute entropy
                entropy = self._compute_entropy(frequencies, n, block_size)
                ApEn[block_size - m] = entropy

        # Calculate test statistics
        apen = ApEn[0] - ApEn[1]
        chi_squared = 2.0 * n * (np.log(2) - apen)
        p_value = gammaincc(2 ** (m - 1), chi_squared / 2.0)

        # Prepare statistics
        stats = {
            'n': n,
            'm': m,
            'chi_squared': chi_squared,
            'phi_m': ApEn[0],
            'phi_m_plus_1': ApEn[1],
            'apen': apen,
            'log2': np.log(2),
        }

        return {
            'success': bool(p_value >= self.significance_level),
            'p_value': float(p_value),
            'statistics': stats,
        }


@jit(nopython=True, cache=True)
def _compute_frequency_jit(sequence, block_size):
    """Compute frequencies of all possible patterns - JIT optimized"""
    seq_length = len(sequence)
    power_len = 2 ** (block_size + 1) - 1
    pattern_counts = np.zeros(power_len, dtype=np.uint32)

    for i in range(seq_length):
        pattern = 1  # Start with 1 as in the C implementation
        for j in range(block_size):
            pattern <<= 1
            idx = (i + j) % seq_length
            if sequence[idx] == 1:
                pattern += 1
        pattern_counts[pattern - 1] += 1

    return pattern_counts, power_len


@jit(nopython=True, cache=True)
def _compute_entropy_jit(frequencies, seq_length, block_size):
    """Compute entropy value for given frequencies - JIT optimized"""
    start_idx = 2 ** block_size - 1
    end_idx = 2 ** (block_size + 1) - 1
    valid_frequencies = frequencies[start_idx:end_idx]

    entropy_sum = 0.0
    for freq in valid_frequencies:
        if freq > 0:
            ratio = freq / seq_length
            entropy_sum += freq * np.log(ratio)

    return entropy_sum / seq_length