import numpy as np
from scipy.special import gammaincc
from pathlib import Path


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
        self.name = "Approximate Entropy Test"

    def _convert_to_binary_list(
        self, input_data: str | bytes | list[int] | np.ndarray
    ) -> np.ndarray:
        """Convert various input formats to a numpy array of integers (0s and 1s)"""
        if isinstance(input_data, str):
            return np.array([int(bit) for bit in input_data])
        if isinstance(input_data, bytes):
            binary_str = "".join(format(byte, "08b") for byte in input_data)
            return np.array([int(bit) for bit in binary_str])
        if isinstance(input_data, (list, np.ndarray)):
            return np.array(input_data)
        raise ValueError("Unsupported input format")

    def _compute_frequency(
        self, sequence: np.ndarray, block_size: int
    ) -> tuple[np.ndarray, int]:
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
        seq_length = len(sequence)
        # Initialize array for pattern counts
        power_len = 2 ** (block_size + 1) - 1
        pattern_counts = np.zeros(power_len, dtype=np.uint32)

        # Count overlapping patterns
        for i in range(seq_length):
            # Build pattern value using bits
            pattern = 1  # Start with 1 as in the C implementation
            for j in range(block_size):
                pattern <<= 1
                if sequence[(i + j) % seq_length] == 1:
                    pattern += 1
            pattern_counts[pattern - 1] += 1

        return pattern_counts, power_len

    def _compute_entropy(
        self, frequencies: np.ndarray, seq_length: int, block_size: int
    ) -> float:
        """
        Compute entropy value for given frequencies

        Args:
            frequencies: Array of pattern frequencies
            seq_length: Length of the sequence
            block_size: Size of blocks used

        Returns:
            float: Computed entropy value
        """
        # Get frequencies for complete patterns
        start_idx = 2**block_size - 1
        end_idx = 2 ** (block_size + 1) - 1
        valid_frequencies = frequencies[start_idx:end_idx]

        # Compute entropy for non-zero frequencies
        non_zero_freq = valid_frequencies[valid_frequencies > 0]
        if len(non_zero_freq) == 0:
            return 0.0

        # Calculate entropy sum
        return (
            np.sum(non_zero_freq * np.log(non_zero_freq / seq_length)) / seq_length
        )


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
                "success": False,
                "error": (
                    f"Block size {m} exceeds recommended value of "
                    f"{max(1, int(np.log2(n) - 5))}. Results may be inaccurate."
                ),
                "statistics": {"n": n, "m": m},
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
            "n": n,
            "m": m,
            "chi_squared": chi_squared,
            "phi_m": ApEn[0],
            "phi_m_plus_1": ApEn[1],
            "apen": apen,
            "log2": np.log(2),
        }

        return {
            "success": bool(p_value >= self.significance_level),
            "p_value": float(p_value),
            "statistics": stats,
        }

    def test_file(self, file_path: str | Path) -> dict:
        """Run the Approximate Entropy Test on a file"""
        with Path.open(file_path, "rb") as f:
            data = f.read()
        return self.test(data)


def format_test_report(test_results: dict) -> str:
    """Format test results as a readable report"""
    if "error" in test_results:
        return (
            "\n\t\t\tAPPROXIMATE ENTROPY TEST\n"
            "\t\t--------------------------------------------\n"
            f"ERROR: {test_results['error']}\n"
        )

    stats = test_results["statistics"]
    status = "SUCCESS" if test_results["success"] else "FAILURE"

    report = [
        "\n\t\t\tAPPROXIMATE ENTROPY TEST",
        "\t\t--------------------------------------------",
        "\t\tCOMPUTATIONAL INFORMATION:",
        "\t\t--------------------------------------------",
        f"\t\t(a) m (block length)    = {stats['m']}",
        f"\t\t(b) n (sequence length) = {stats['n']}",
        f"\t\t(c) Chi^2               = {stats['chi_squared']:.6f}",
        f"\t\t(d) Phi(m)              = {stats['phi_m']:.6f}",
        f"\t\t(e) Phi(m+1)            = {stats['phi_m_plus_1']:.6f}",
        f"\t\t(f) ApEn                = {stats['apen']:.6f}",
        f"\t\t(g) Log(2)              = {stats['log2']:.6f}",
        "\t\t--------------------------------------------",
    ]

    if stats["m"] > int(np.log2(stats["n"]) - 5):
        report.extend(
            [
                f"\t\tNote: The blockSize = {stats['m']} exceeds recommended value "
                f"of {max(1, int(np.log2(stats['n']) - 5))}",
                "\t\tResults are inaccurate!",
                "\t\t--------------------------------------------",
            ]
        )

    report.append(f"{status}\t\tp_value = {test_results['p_value']:.6f}\n")

    return "\n".join(report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NIST Approximate Entropy Test")
    parser.add_argument("file", type=str, help="Path to the binary file to test")
    parser.add_argument(
        "--block-length",
        type=int,
        default=10,
        help="Length m of each block (default: 10)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01, help="Significance level (default: 0.01)"
    )

    args = parser.parse_args()

    # Run test
    test = ApproximateEntropyTest(
        block_length=args.block_length, significance_level=args.alpha
    )
    results = test.test_file(args.file)

    # Print report
