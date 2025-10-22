import numpy as np
from scipy.stats import norm
from pathlib import Path


class CumulativeSumsTest:
    """
    Implementation of NIST's Cumulative Sums (Cusums) Test

    This test determines whether the cumulative sum of the partial sequences occurring
    in the tested sequence is too large or too small relative to the expected behavior
    of that cumulative sum for random sequences. The cumulative sum may be considered
    as a random walk. For a random sequence, the excursions of the random walk should
    be near zero.
    """

    def __init__(self, significance_level: float = 0.01):
        """
        Initialize the Cumulative Sums Test

        Args:
            significance_level: The significance level. Default is 0.01 (1%)
        """
        self.significance_level = significance_level
        self.name = 'Cumulative Sums (Cusums) Test'

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

    def _compute_p_value(self, z: int, n: int) -> float:
        """
        Compute p-value for given maximum excursion and sequence length

        Args:
            z: Maximum excursion (absolute value)
            n: Sequence length

        Returns:
            float: Computed p-value
        """
        # Compute ranges for both sums
        k_range1 = np.arange((-n / z + 1) // 4, (n / z - 1) // 4 + 1)
        k_range2 = np.arange((-n / z - 3) // 4, (n / z - 1) // 4 + 1)

        # Vectorized computation of both sums
        sum1 = np.sum(norm.cdf((4 * k_range1 + 1) * z / np.sqrt(n)) - norm.cdf((4 * k_range1 - 1) * z / np.sqrt(n)))

        sum2 = np.sum(norm.cdf((4 * k_range2 + 3) * z / np.sqrt(n)) - norm.cdf((4 * k_range2 + 1) * z / np.sqrt(n)))

        return 1.0 - sum1 + sum2

    def test(self, binary_data: str | bytes | list[int] | np.ndarray) -> dict:
        """
        Run the Cumulative Sums Test

        Args:
            binary_data: The sequence to test

        Returns:
            dict: Test results containing:
                - success: Boolean indicating if test was passed
                - p_values: The two calculated p-values (forward and reverse)
                - statistics: Additional test statistics
        """
        # Convert input to binary sequence
        sequence = self._convert_to_binary_list(binary_data)
        n = len(sequence)

        # Convert zeros to -1, ones stay 1
        sequence = 2 * sequence - 1

        # Forward Cumulative Sums Test
        S = np.cumsum(sequence)
        sup = np.maximum.accumulate(S)
        inf = np.minimum.accumulate(S)

        # Maximum absolute excursion (forward)
        z_forward = max(*sup, -min(inf))

        # Reverse sequence for backward test
        S_rev = np.cumsum(sequence[::-1])
        sup_rev = np.maximum.accumulate(S_rev)
        inf_rev = np.minimum.accumulate(S_rev)

        # Maximum absolute excursion (reverse)
        z_reverse = max(*sup_rev, -min(inf_rev))

        # Calculate p-values
        p_value_forward = self._compute_p_value(z_forward, n)
        p_value_reverse = self._compute_p_value(z_reverse, n)

        # Prepare statistics
        stats = {
            'n': n,
            'forward_max_excursion': int(z_forward),
            'reverse_max_excursion': int(z_reverse),
        }

        return {
            'success': bool(p_value_forward >= self.significance_level and p_value_reverse >= self.significance_level),
            'p_value_forward': float(p_value_forward),
            'p_value_reverse': float(p_value_reverse),
            'statistics': stats,
        }

    def test_file(self, file_path: str | Path) -> dict:
        """Run the Cumulative Sums Test on a file"""
        with Path.open(file_path, 'rb') as f:
            data = f.read()
        return self.test(data)


def format_test_report(test_results: dict) -> str:
    """Format test results as a readable report"""
    if 'error' in test_results:
        return (
            '\n\t\tCUMULATIVE SUMS TEST\n'
            '\t\t----------------------------------------\n'
            f'ERROR: {test_results["error"]}\n'
        )

    stats = test_results['statistics']
    status_forward = 'SUCCESS' if test_results['p_value_forward'] >= 0.01 else 'FAILURE'  # noqa
    status_reverse = 'SUCCESS' if test_results['p_value_reverse'] >= 0.01 else 'FAILURE'  # noqa

    report = [
        '\n\t\t      CUMULATIVE SUMS (FORWARD) TEST',
        '\t\t-------------------------------------------',
        '\t\tCOMPUTATIONAL INFORMATION:',
        '\t\t-------------------------------------------',
        f'\t\t(a) The maximum partial sum = {stats["forward_max_excursion"]}',
        '\t\t-------------------------------------------',
    ]

    if not (0 <= test_results['p_value_forward'] <= 1):
        report.append('\t\tWARNING:  P_VALUE IS OUT OF RANGE')

    report.extend(
        [
            f'{status_forward}\t\tp_value = {test_results["p_value_forward"]:.6f}\n',
            '\t\t      CUMULATIVE SUMS (REVERSE) TEST',
            '\t\t-------------------------------------------',
            '\t\tCOMPUTATIONAL INFORMATION:',
            '\t\t-------------------------------------------',
            f'\t\t(a) The maximum partial sum = {stats["reverse_max_excursion"]}',
            '\t\t-------------------------------------------',
        ]
    )

    if not (0 <= test_results['p_value_reverse'] <= 1):
        report.append('\t\tWARNING:  P_VALUE IS OUT OF RANGE')

    report.append(f'{status_reverse}\t\tp_value = {test_results["p_value_reverse"]:.6f}\n')

    return '\n'.join(report)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NIST Cumulative Sums Test')
    parser.add_argument('file', type=str, help='Path to the binary file to test')
    parser.add_argument('--alpha', type=float, default=0.01, help='Significance level (default: 0.01)')

    args = parser.parse_args()

    # Run test
    test = CumulativeSumsTest(significance_level=args.alpha)
    results = test.test_file(args.file)

    # Print report
