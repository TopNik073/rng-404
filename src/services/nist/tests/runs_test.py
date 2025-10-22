import numpy as np
from scipy.special import erfc
from pathlib import Path


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
        tau = 2.0 / np.sqrt(n)
        if abs(pi - 0.5) >= tau:
            return {
                'success': False,
                'p_value': 0.0,
                'statistics': {
                    'n': n,
                    'proportion_ones': pi,
                    'tau': tau,
                    'criteria_met': False,
                    'error': 'Pi estimator criteria not met',
                },
            }

        # Count runs
        V_n_obs = self._count_runs(sequence)

        # Calculate test statistic
        p = 0.5  # Expected proportion under randomness assumption
        exp_runs = 2 * n * p * (1 - p)  # Expected number of runs
        std_dev = 2 * np.sqrt(n) * p * (1 - p)  # Standard deviation term

        erfc_arg = abs(V_n_obs - exp_runs) / std_dev
        p_value = erfc(erfc_arg)

        # Ensure p-value is valid
        if p_value < 0 or p_value > 1:
            p_value = 0 if p_value < 0 else 1

        # Prepare statistics
        stats = {
            'n': n,
            'proportion_ones': pi,
            'tau': tau,
            'criteria_met': True,
            'observed_runs': V_n_obs,
            'expected_runs': exp_runs,
            'test_statistic': erfc_arg,
        }

        return {
            'success': bool(p_value >= self.significance_level),
            'p_value': float(p_value),
            'statistics': stats,
        }

    def test_file(self, file_path: str | Path) -> dict:
        """
        Run the Runs Test on a file

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

    if not stats['criteria_met']:
        report = [
            'RUNS TEST',
            '-' * 45,
            'PI ESTIMATOR CRITERIA NOT MET!',
            f'Pi = {stats["proportion_ones"]:.6f}',
            f'Tau = {stats["tau"]:.6f}',
            '-' * 45,
            'FAILURE',
            'p_value = 0.000000\n',
        ]
    else:
        report = [
            'RUNS TEST',
            '-' * 45,
            'COMPUTATIONAL INFORMATION:',
            '-' * 45,
            f'(a) Pi (proportion of ones)      = {stats["proportion_ones"]:.6f}',
            f'(b) Tau (threshold)              = {stats["tau"]:.6f}',
            f'(c) Total number of runs         = {stats["observed_runs"]}',
            f'(d) Expected number of runs      = {stats["expected_runs"]:.6f}',
            f'(e) Test statistic               = {stats["test_statistic"]:.6f}',
            '-' * 45,
            'SUCCESS' if test_results['success'] else 'FAILURE',
            f'p_value = {test_results["p_value"]:.6f}\n',
        ]

    return '\n'.join(report)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NIST Runs Test')
    parser.add_argument('file', type=str, help='Path to the binary file to test')
    parser.add_argument('--alpha', type=float, default=0.01, help='Significance level (default: 0.01)')

    args = parser.parse_args()

    # Run test
    test = RunsTest(significance_level=args.alpha)
    results = test.test_file(args.file)

    # Print report
