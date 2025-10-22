import numpy as np
from scipy.special import gammaincc
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BlockConfig:
    """Configuration for different sequence lengths"""

    K: int  # Number of degrees of freedom
    M: int  # Block length
    V: list[int]  # Values for run length categories
    pi: list[float]  # Theoretical probabilities


class LongestRunsTest:
    """
    Implementation of NIST's Longest Runs of Ones in a Block Test

    This test focuses on the longest run of ones within M-bit blocks. The purpose of
    this test is to determine whether the length of the longest run of ones within
    the tested sequence is consistent with the length of the longest run of ones that
    would be expected in a random sequence.
    """

    def __init__(self, significance_level: float = 0.01):
        """
        Initialize the Longest Runs Test

        Args:
            significance_level: The significance level. Default is 0.01 (1%)
        """
        self.significance_level = significance_level
        self.name = 'Longest Runs of Ones Test'

        # Define configurations for different sequence lengths
        self.configs = {
            'short': BlockConfig(K=3, M=8, V=[1, 2, 3, 4], pi=[0.21484375, 0.3671875, 0.23046875, 0.1875]),
            'medium': BlockConfig(
                K=5,
                M=128,
                V=[4, 5, 6, 7, 8, 9],
                pi=[
                    0.1174035788,
                    0.242955959,
                    0.249363483,
                    0.17517706,
                    0.102701071,
                    0.112398847,
                ],
            ),
            'long': BlockConfig(
                K=6,
                M=10000,
                V=[10, 11, 12, 13, 14, 15, 16],
                pi=[0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727],
            ),
        }

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

    def _get_config(self, n: int) -> BlockConfig:
        """Get the appropriate configuration based on sequence length"""
        if n < 128:  # noqa
            raise ValueError(f'Sequence length {n} is too short (minimum 128)')
        if n < 6272:  # noqa
            return self.configs['short']
        if n < 750000:  # noqa
            return self.configs['medium']
        return self.configs['long']

    def _find_longest_run_in_block(self, block: np.ndarray) -> int:
        """Find the longest run of ones in a block"""
        # Convert to string and split by zeros
        runs = ''.join(map(str, block)).split('0')
        # Return length of longest run
        return max(len(run) for run in runs) if runs else 0

    def _count_frequencies(self, sequence: np.ndarray, config: BlockConfig) -> tuple[list[int], int]:
        """Count frequencies of different run lengths in blocks"""
        N = len(sequence) // config.M  # Number of blocks
        nu = np.zeros(config.K + 1, dtype=int)

        # Process each block
        for i in range(N):
            block = sequence[i * config.M : (i + 1) * config.M]
            longest_run = self._find_longest_run_in_block(block)

            # Categorize the longest run
            if longest_run < config.V[0]:
                nu[0] += 1
            elif longest_run > config.V[-1]:
                nu[-1] += 1
            else:
                for j, v in enumerate(config.V):
                    if longest_run == v:
                        nu[j] += 1
                        break

        return nu, N

    def test(self, binary_data: str | bytes | list[int] | np.ndarray) -> dict:
        """
        Run the Longest Runs Test

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

        # Get configuration based on sequence length
        try:
            config = self._get_config(n)
        except ValueError as e:
            return {
                'success': False,
                'p_value': 0.0,
                'statistics': {'error': str(e), 'n': n},
            }

        # Count frequencies
        nu, N = self._count_frequencies(sequence, config)

        # Calculate chi-square statistic
        expected = np.array(config.pi) * N
        chi_squared = np.sum((nu - expected) ** 2 / expected)

        # Calculate p-value
        p_value = gammaincc(config.K / 2.0, chi_squared / 2.0)

        # Ensure p-value is valid
        if p_value < 0 or p_value > 1:
            p_value = 0 if p_value < 0 else 1

        # Prepare statistics
        stats = {
            'n': n,
            'block_size': config.M,
            'num_blocks': N,
            'chi_squared': chi_squared,
            'degrees_of_freedom': config.K,
            'frequencies': nu.tolist(),
            'expected_frequencies': expected.tolist(),
            'run_length_categories': config.V,
        }

        return {
            'success': bool(p_value >= self.significance_level),
            'p_value': float(p_value),
            'statistics': stats,
        }

    def test_file(self, file_path: str | Path) -> dict:
        """Run the Longest Runs Test on a file"""
        with Path.open(file_path, 'rb') as f:
            data = f.read()
        return self.test(data)


def format_test_report(test_results: dict) -> str:
    """Format test results as a readable report"""
    stats = test_results['statistics']

    if 'error' in stats:
        return f'\nLONGEST RUNS OF ONES TEST\n{"-" * 45}\nERROR: {stats["error"]}\n'

    report = [
        '\nLONGEST RUNS OF ONES TEST',
        '-' * 45,
        'COMPUTATIONAL INFORMATION:',
        '-' * 45,
        f'(a) N (# of blocks)     = {stats["num_blocks"]}',
        f'(b) M (Block length)    = {stats["block_size"]}',
        f'(c) Chi^2               = {stats["chi_squared"]:.6f}',
        '-' * 45,
        '      F R E Q U E N C Y',
        '-' * 45,
    ]

    # Add frequency table header based on block size
    if stats['block_size'] == 8:  # noqa
        report.append('  <=1     2     3    >=4   P-value  Assignment')
    elif stats['block_size'] == 128:  # noqa
        report.append('  <=4  5  6  7  8  >=9 P-value  Assignment')
    else:
        report.append('  <=10  11  12  13  14  15 >=16 P-value  Assignment')

    # Add frequencies
    freq_str = '  ' + '  '.join(f'{f:3d}' for f in stats['frequencies'])
    report.append(freq_str)

    # Add result
    report.extend(
        [
            f'{"SUCCESS" if test_results["success"] else "FAILURE"}',
            f'p_value = {test_results["p_value"]:.6f}\n',
        ]
    )

    return '\n'.join(report)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NIST Longest Runs of Ones Test')
    parser.add_argument('file', type=str, help='Path to the binary file to test')
    parser.add_argument('--alpha', type=float, default=0.01, help='Significance level (default: 0.01)')

    args = parser.parse_args()

    # Run test
    test = LongestRunsTest(significance_level=args.alpha)
    results = test.test_file(args.file)

    # Print report
