import numpy as np
from scipy.special import gammaincc
from typing import Union, List, Dict, Tuple
from pathlib import Path


class LinearComplexityTest:
    """
    Implementation of NIST's Linear Complexity Test

    This test determines whether or not the sequence is complex enough to be considered
    random. Random sequences are characterized by longer linear feedback shift registers (LFSRs).
    An LFSR that is too short implies non-randomness.
    """

    def __init__(self, block_size: int = 500, significance_level: float = 0.01):
        """
        Initialize the Linear Complexity Test

        Args:
            block_size: The length M of each block. Default is 500
            significance_level: The significance level (α). Default is 0.01 (1%)
        """
        self.block_size = block_size
        self.significance_level = significance_level
        self.name = "Linear Complexity Test"

        # Number of degrees of freedom (categories)
        self.K = 6

        # Probabilities for the K+1 degrees of freedom (from NIST implementation)
        self.pi = np.array(
            [0.01047, 0.03125, 0.12500, 0.50000, 0.25000, 0.06250, 0.020833]
        )

    def _convert_to_binary_list(
        self, input_data: Union[str, bytes, List[int], np.ndarray]
    ) -> np.ndarray:
        """Convert various input formats to a numpy array of integers (0s and 1s)"""
        if isinstance(input_data, str):
            return np.array([int(bit) for bit in input_data])
        elif isinstance(input_data, bytes):
            binary_str = "".join(format(byte, "08b") for byte in input_data)
            return np.array([int(bit) for bit in binary_str])
        elif isinstance(input_data, (list, np.ndarray)):
            return np.array(input_data)
        else:
            raise ValueError("Unsupported input format")

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
        n = len(block)
        c = np.zeros(n, dtype=int)  # Connection polynomial
        b = np.zeros(n, dtype=int)  # Previous connection polynomial
        c[0] = 1
        b[0] = 1

        L = 0  # Length of LFSR
        m = -1  # Index of last update
        N = 0  # Number of bits processed

        while N < n:
            # Compute discrepancy
            d = int(block[N])
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

    def test(self, binary_data: Union[str, bytes, List[int], np.ndarray]) -> dict:
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
                "success": False,
                "error": f"Insufficient data. Need at least {self.block_size} bits.",
                "statistics": {"n": n, "block_size": self.block_size},
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
            # Mean = M/2 + (9 + (M+1)%2)/36 - 1/2^M * (M/3 + 2/9)
            M = self.block_size
            parity = (M + 1) % 2
            sign = 1 if parity == 0 else -1
            mean = (
                M / 2.0
                + (9.0 + sign) / 36.0
                - (1.0 / pow(2, M)) * (M / 3.0 + 2.0 / 9.0)
            )

            # Calculate T value
            sign = 1 if M % 2 == 0 else -1
            T = sign * (L - mean) + 2.0 / 9.0

            # Assign to appropriate bin
            if T <= -2.5:
                nu[0] += 1
            elif T <= -1.5:
                nu[1] += 1
            elif T <= -0.5:
                nu[2] += 1
            elif T <= 0.5:
                nu[3] += 1
            elif T <= 1.5:
                nu[4] += 1
            elif T <= 2.5:
                nu[5] += 1
            else:
                nu[6] += 1

        # Calculate chi-square statistic
        chi_squared = np.sum((nu - N * self.pi) ** 2 / (N * self.pi))

        # Calculate p-value
        p_value = gammaincc(self.K / 2.0, chi_squared / 2.0)

        # Prepare statistics
        stats = {
            "n": n,
            "block_size": self.block_size,
            "num_blocks": N,
            "chi_squared": chi_squared,
            "frequencies": nu.tolist(),
            "expected_frequencies": (N * self.pi).tolist(),
            "bits_discarded": n % self.block_size,
        }

        return {
            "success": bool(p_value >= self.significance_level),
            "p_value": float(p_value),
            "statistics": stats,
        }

    def test_file(self, file_path: Union[str, Path]) -> dict:
        """Run the Linear Complexity Test on a file"""
        with open(file_path, "rb") as f:
            data = f.read()
        return self.test(data)


def format_test_report(test_results: dict) -> str:
    """Format test results as a readable report"""
    if "error" in test_results:
        return (
            "\n\t\tLINEAR COMPLEXITY TEST\n"
            "-----------------------------------------------------\n"
            f"ERROR: {test_results['error']}\n"
        )

    stats = test_results["statistics"]

    report = [
        "\n-----------------------------------------------------",
        "\tL I N E A R  C O M P L E X I T Y",
        "-----------------------------------------------------",
        f"\tM (substring length)     = {stats['block_size']}",
        f"\tN (number of substrings) = {stats['num_blocks']}",
        "-----------------------------------------------------",
        "        F R E Q U E N C Y",
        "-----------------------------------------------------",
        "  C0   C1   C2   C3   C4   C5   C6    CHI2    P-value",
        "-----------------------------------------------------",
    ]

    # Format frequencies
    freq_str = " ".join(f"{int(f):4d}" for f in stats["frequencies"])
    report.append(
        f"{freq_str} {stats['chi_squared']:9.6f} {test_results['p_value']:9.6f}"
    )

    if stats["bits_discarded"] > 0:
        report.append(f"\tNote: {stats['bits_discarded']} bits were discarded!")

    return "\n".join(report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NIST Linear Complexity Test")
    parser.add_argument("file", type=str, help="Path to the binary file to test")
    parser.add_argument(
        "--block-size",
        type=int,
        default=500,
        help="Length M of each block (default: 500)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01, help="Significance level (default: 0.01)"
    )

    args = parser.parse_args()

    # Run test
    test = LinearComplexityTest(
        block_size=args.block_size, significance_level=args.alpha
    )
    results = test.test_file(args.file)

    # Print report
    print(format_test_report(results))
