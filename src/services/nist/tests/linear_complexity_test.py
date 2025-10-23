import numpy as np
from scipy.special import gammaincc
from numba import jit


class LinearComplexityTest:
    """
    Implementation of NIST's Linear Complexity Test

    This test determines whether the sequence is complex enough to be considered
    random. Random sequences are characterized by longer linear feedback shift registers (LFSRs).
    An LFSR that is too short implies non-randomness.
    """

    def __init__(self, block_size: int = 500, significance_level: float = 0.01):
        """
        Initialize the Linear Complexity Test

        Args:
            block_size: The length M of each block. Default is 500
            significance_level: The significance level. Default is 0.01 (1%)
        """
        self.block_size = block_size
        self.significance_level = significance_level
        self.name = 'Linear Complexity Test'

        # Number of degrees of freedom (categories)
        self.K = 6

        # Probabilities for the K+1 degrees of freedom (from NIST implementation)
        self.pi = np.array([0.01047, 0.03125, 0.12500, 0.50000, 0.25000, 0.06250, 0.020833])

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

    def _berlekamp_massey(self, block: np.ndarray) -> int:
        """
        Implementation of the Berlekamp-Massey Algorithm

        This algorithm determines the minimal polynomial of a linearly recurrent sequence,
        which is equivalent to finding the shortest linear feedback shift register (LFSR)
        for a sequence.

        Args:
            block: A block of bits to analyze

        Returns:
            int: The linear complexity (length of the shortest LFSR) for the sequence
        """
        block_int32 = block.astype(np.int32)
        return _berlekamp_massey_jit(block_int32)

    def test(self, binary_data: str | bytes | list[int] | np.ndarray) -> dict:
        """
        Run the Linear Complexity Test

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

        # Calculate number of blocks
        N = n // self.block_size
        if N == 0:
            return {
                'success': False,
                'error': f'Insufficient data. Need at least {self.block_size} bits.',
                'statistics': {'n': n, 'block_size': self.block_size},
            }

        # Initialize frequency array for the K+1 bins
        nu = np.zeros(self.K + 1)

        # Process each block
        for i in range(N):
            # Extract block
            block = sequence[i * self.block_size : (i + 1) * self.block_size]

            # Calculate linear complexity for the block
            L = self._berlekamp_massey(block)

            # Calculate expected mean value
            M = self.block_size
            parity = (M + 1) % 2
            sign = 1 if parity == 0 else -1
            mean = M / 2.0 + (9.0 + sign) / 36.0 - (1.0 / pow(2, M)) * (M / 3.0 + 2.0 / 9.0)

            # Calculate T value
            sign = 1 if M % 2 == 0 else -1
            T = sign * (L - mean) + 2.0 / 9.0

            # Assign to appropriate bin
            if T <= -2.5:  # noqa
                nu[0] += 1
            elif T <= -1.5:  # noqa
                nu[1] += 1
            elif T <= -0.5:  # noqa
                nu[2] += 1
            elif T <= 0.5:  # noqa
                nu[3] += 1
            elif T <= 1.5:  # noqa
                nu[4] += 1
            elif T <= 2.5:  # noqa
                nu[5] += 1
            else:
                nu[6] += 1

        # Calculate chi-square statistic
        chi_squared = np.sum((nu - N * self.pi) ** 2 / (N * self.pi))

        # Calculate p-value
        p_value = gammaincc(self.K / 2.0, chi_squared / 2.0)

        # Prepare statistics
        stats = {
            'n': n,
            'block_size': self.block_size,
            'num_blocks': N,
            'chi_squared': chi_squared,
            'frequencies': nu.tolist(),
            'expected_frequencies': (N * self.pi).tolist(),
            'bits_discarded': n % self.block_size,
        }

        return {
            'success': bool(p_value >= self.significance_level),
            'p_value': float(p_value),
            'statistics': stats,
        }


@jit(nopython=True, cache=True)
def _berlekamp_massey_jit(block):
    """
    Implementation of the Berlekamp-Massey Algorithm - JIT optimized

    Args:
        block: A block of bits to analyze

    Returns:
        int: The linear complexity (length of the shortest LFSR) for the sequence
    """
    n = len(block)
    c = np.zeros(n, dtype=np.int32)  # Connection polynomial
    b = np.zeros(n, dtype=np.int32)  # Previous connection polynomial
    c[0] = 1
    b[0] = 1

    L = 0  # Length of LFSR
    m = -1  # Index of last update
    N = 0  # Number of bits processed

    while n > N:
        # Compute discrepancy
        d = block[N]
        for i in range(1, L + 1):
            d ^= c[i] & block[N - i]  # XOR operation

        if d == 1:  # If there is a discrepancy
            t = c.copy()
            for j in range(n - N + m):
                if b[j] == 1:
                    c[j + N - m] ^= 1  # Update using XOR

            if L <= N / 2:
                L = N + 1 - L
                m = N
                b = t.copy()

        N += 1

    return L