import numpy as np
from scipy.fft import rfft
from scipy.special import erfc
from pathlib import Path


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

    def test_file(self, file_path: str | Path) -> dict:
        """
        Run the DFT Test on a file

        Args:
            file_path: Path to the file containing binary data

        Returns:
            dict: Test results (same as test() method)
        """
        with Path.open(file_path, 'rb') as f:
            data = f.read()
        return self.test(data)


def format_test_report(test_results: dict) -> str:
    """
    Format test results as a readable report

    Args:
        test_results: Dictionary containing test results

    Returns:
        str: Formatted report
    """
    stats = test_results['statistics']

    report = [
        '\nDISCRETE FOURIER TRANSFORM TEST',
        '-' * 45,
        'COMPUTATIONAL INFORMATION:',
        '-' * 45,
        f'(a) Percentile      = {stats["percentile"]:.6f}',
        f'(b) N_l            = {stats["N_l"]:.6f}',
        f'(c) N_o            = {stats["N_o"]:.6f}',
        f'(d) d              = {stats["d"]:.6f}',
        f'(e) Threshold      = {stats["threshold"]:.6f}',
        f'(f) Peaks analyzed = {stats["peaks_total"]}',
        '-' * 45,
        f'{"SUCCESS" if test_results["success"] else "FAILURE"}',
        f'p_value = {test_results["p_value"]:.6f}\n',
    ]

    return '\n'.join(report)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NIST Discrete Fourier Transform Test')
    parser.add_argument('file', type=str, help='Path to the binary file to test')
    parser.add_argument('--alpha', type=float, default=0.01, help='Significance level (default: 0.01)')

    args = parser.parse_args()

    # Run test
    test = DiscreteFourierTransformTest(significance_level=args.alpha)
    results = test.test_file(args.file)

    # Print report
