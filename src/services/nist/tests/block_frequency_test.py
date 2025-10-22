import numpy as np
from scipy.special import gammaincc
from pathlib import Path


class BlockFrequencyTest:
    """
    Implementation of NIST's Block Frequency Test

    This test divides the input sequence into blocks and analyzes the proportion
    of ones in each block. The purpose of this test is to determine whether the
    frequency of ones in an M-bit block is approximately M/2, as would be expected
    under an assumption of randomness.
    """

    def __init__(self, block_size: int = 128, significance_level: float = 0.01):
        """
        Initialize the Block Frequency Test

        Args:
            block_size: The size of each block (M). Default is 128
            significance_level: The significance level. Default is 0.01 (1%)
        """
        self.block_size = block_size
        self.significance_level = significance_level
        self.name = 'Block Frequency Test'

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
        Run the Block Frequency Test

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

        # Check if sequence length is sufficient
        if n < self.block_size:
            raise ValueError(f'Input sequence length ({n}) is too short for block size {self.block_size}')

        # Calculate number of blocks and truncate sequence if necessary
        N = n // self.block_size
        truncated_seq = sequence[: N * self.block_size]
        discarded_bits = n - N * self.block_size

        # Reshape sequence into blocks
        blocks = truncated_seq.reshape(N, self.block_size)

        # Calculate proportion of ones in each block
        pi = np.sum(blocks, axis=1) / self.block_size

        # Calculate chi-squared statistic
        v = pi - 0.5
        chi_squared = 4.0 * self.block_size * np.sum(v * v)

        # Calculate p-value using incomplete gamma function
        p_value = gammaincc(N / 2.0, chi_squared / 2.0)

        # Prepare statistics
        stats = {
            'n': n,
            'block_size': self.block_size,
            'num_blocks': N,
            'discarded_bits': discarded_bits,
            'chi_squared': chi_squared,
            'block_frequencies': pi.tolist(),  # Convert to list for JSON serialization
        }

        return {
            'success': bool(p_value >= self.significance_level),
            'p_value': float(p_value),
            'statistics': stats,
        }

    def test_file(self, file_path: str | Path) -> dict:
        """
        Run the Block Frequency Test on a file

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
        'BLOCK FREQUENCY TEST',
        '-' * 45,
        'COMPUTATIONAL INFORMATION:',
        '-' * 45,
        f'(a) Chi^2             = {stats["chi_squared"]:.6f}',
        f'(b) # of blocks       = {stats["num_blocks"]}',
        f'(c) Block length      = {stats["block_size"]}',
        f'(d) Sequence length   = {stats["n"]}',
        f'(e) Discarded bits    = {stats["discarded_bits"]}',
        '-' * 45,
        'SUCCESS' if test_results['success'] else 'FAILURE',
        f'p_value = {test_results["p_value"]:.6f}\n',
    ]

    return '\n'.join(report)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NIST Block Frequency Test')
    parser.add_argument('file', type=str, help='Path to the binary file to test')
    parser.add_argument('--block-size', type=int, default=128, help='Size of each block (default: 128)')
    parser.add_argument('--alpha', type=float, default=0.01, help='Significance level (default: 0.01)')

    args = parser.parse_args()

    # Run test
    test = BlockFrequencyTest(block_size=args.block_size, significance_level=args.alpha)
    results = test.test_file(args.file)

    # Print report
