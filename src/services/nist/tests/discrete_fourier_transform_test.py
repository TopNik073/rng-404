import numpy as np
from scipy.fft import rfft
from scipy.special import erfc


class DiscreteFourierTransformTest:
    """
    Implementation of NIST's Discrete Fourier Transform (Spectral) Test

    This test checks for periodic features in the tested sequence that would
    indicate a deviation from randomness. The focus is on the peak heights in
    the Discrete Fourier Transform of the sequence. The purpose of this test
    is to detect periodic patterns in the original sequence that would indicate
    a deviation from randomness.
    """

    def __init__(self, significance_level: float = 0.01):
        """
        Initialize the DFT Test

        Args:
            significance_level: The significance level. Default is 0.01 (1%)
        """
        self.significance_level = significance_level
        self.name = 'Discrete Fourier Transform (Spectral) Test'

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

    def test(self, binary_data: str | bytes | list[int] | np.ndarray) -> dict:
        """
        Run the Discrete Fourier Transform Test

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

        # Convert zeros and ones to -1 and +1
        X = 2 * sequence - 1

        # Apply FFT
        S = np.abs(rfft(X))

        # Get magnitudes (excluding DC component)
        magnitudes = S[1:]

        # Compute threshold
        T = np.sqrt(2.995732274 * n)  # Theoretical threshold value

        # Count values below threshold
        N_l = np.sum(magnitudes < T)

        # Calculate statistics
        N_o = 0.95 * n / 2  # Expected number of peaks less than T
        d = (N_l - N_o) / np.sqrt(n / 4 * 0.95 * 0.05)
        p_value = erfc(abs(d) / np.sqrt(2))

        # Calculate percentile
        percentile = (N_l / (n / 2)) * 100

        # Prepare statistics
        stats = {
            'n': n,
            'threshold': T,
            'percentile': percentile,
            'N_l': N_l,  # Number of peaks below threshold
            'N_o': N_o,  # Expected number of peaks below threshold
            'd': d,  # Normalized difference
            'peaks_total': len(magnitudes),
            'peaks_below_threshold': int(N_l),
        }

        return {
            'success': bool(p_value >= self.significance_level),
            'p_value': float(p_value),
            'statistics': stats,
        }
