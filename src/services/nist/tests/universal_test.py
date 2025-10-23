import numpy as np
from scipy.special import erfc
from pathlib import Path


class UniversalTest:
    """
    Implementation of Maurer's "Universal Statistical" Test

    This test focuses on the number of bits between matching patterns. The purpose of
    this test is to detect whether or not the sequence can be significantly compressed
    without loss of information. A significantly compressible sequence is considered
    to be non-random.
    """

    def __init__(self, significance_level: float = 0.01):
        """
        Initialize the Universal Statistical Test

        Args:
            significance_level: The significance level. Default is 0.01 (1%)
        """
        self.significance_level = significance_level
        self.name = 'Universal Statistical Test'

        # Pre-computed expected values for L=6 to L=16 (indexes 0-5 are unused)
        self.expected_value = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                5.2177052,
                6.1962507,
                7.1836656,
                8.1764248,
                9.1723243,
                10.170032,
                11.168765,
                12.168070,
                13.167693,
                14.167488,
                15.167379,
            ]
        )

        # Pre-computed variances for L=6 to L=16 (indexes 0-5 are unused)
        self.variance = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                2.954,
                3.125,
                3.238,
                3.311,
                3.356,
                3.384,
                3.401,
                3.410,
                3.416,
                3.419,
                3.421,
            ]
        )

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

    def _determine_block_size(self, n: int) -> int:
        """
        Determine the appropriate block size (L) based on sequence length

        Args:
            n: Length of the sequence

        Returns:
            int: Appropriate block size L
        """
        L = 5  # Minimum block size
        block_size_thresholds = [
            (387840, 6),
            (904960, 7),
            (2068480, 8),
            (4654080, 9),
            (10342400, 10),
            (22753280, 11),
            (49643520, 12),
            (107560960, 13),
            (231669760, 14),
            (496435200, 15),
            (1059061760, 16),
        ]

        for threshold, size in block_size_thresholds:
            if n >= threshold:
                L = size
            else:
                break

        return L

    def _blocks_to_integers(self, sequence: np.ndarray, L: int) -> np.ndarray:
        """Convert sequence of bits to array of integers using block size L"""
        # Reshape sequence into blocks of L bits
        blocks = sequence.reshape(-1, L)

        # Create powers of 2 array for conversion: [2^(L-1), 2^(L-2), ..., 2^0]
        powers = 2 ** np.arange(L - 1, -1, -1)

        # Convert each block to an integer
        return np.sum(blocks * powers, axis=1)

    def test(self, binary_data: str | bytes | list[int] | np.ndarray) -> dict:
        """
        Run Maurer's Universal Statistical Test

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

        # Determine block size L and parameters Q and K
        L = self._determine_block_size(n)
        Q = 10 * 2**L  # Initialize phase blocks
        K = n // L - Q  # Test phase blocks

        # Validate parameters
        if L < 6 or L > 16 or Q < 10 * 2**L:  # noqa
            return {
                'success': False,
                'error': (f'Invalid parameters: L={L} must be between 6 and 16, and Q={Q} must be >= {10 * 2**L}'),
                'statistics': {'n': n, 'L': L, 'Q': Q, 'K': K},
            }

        # Ensure sequence length is sufficient
        if n < (Q + K) * L:
            return {
                'success': False,
                'error': f'Insufficient sequence length. Need at least {(Q + K) * L} bits.',
                'statistics': {'n': n, 'L': L, 'Q': Q, 'K': K},
            }

        # Initialize table
        p = 2**L
        T = np.zeros(p, dtype=np.int64)

        # Convert sequence to array of integers
        int_sequence = self._blocks_to_integers(sequence[: (Q + K) * L].reshape(-1, L), L)

        # Initialize table with the first Q blocks
        for i in range(Q):
            T[int_sequence[i]] = i + 1

        # Process remaining K blocks
        sum_log = 0.0
        for i in range(Q, Q + K):
            pattern = int_sequence[i]
            distance = i + 1 - T[pattern]
            sum_log += np.log2(distance)
            T[pattern] = i + 1

        # Compute test statistics
        phi = sum_log / K
        c = 0.7 - 0.8 / L + (4 + 32 / L) * K ** (-3 / L) / 15
        sigma = c * np.sqrt(self.variance[L] / K)

        # Calculate p-value
        arg = abs(phi - self.expected_value[L]) / (np.sqrt(2) * sigma)
        p_value = erfc(arg)

        # Prepare statistics
        stats = {
            'n': n,
            'L': L,
            'Q': Q,
            'K': K,
            'sum': sum_log,
            'sigma': sigma,
            'variance': self.variance[L],
            'expected_value': self.expected_value[L],
            'phi': phi,
            'bits_discarded': n - (Q + K) * L,
        }

        return {
            'success': bool(p_value >= self.significance_level),
            'p_value': float(p_value),
            'statistics': stats,
        }
