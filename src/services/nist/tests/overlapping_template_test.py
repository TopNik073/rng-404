import numpy as np
from scipy.special import gammaincc, gammaln
from typing import Union, List, Dict, Tuple
from pathlib import Path


class OverlappingTemplateTest:
    """
    Implementation of NIST's Overlapping Template Matching Test

    This test uses an m-bit window to search for a specific m-bit pattern. The window
    slides forward one bit after each search. The test checks whether the number
    of found matches is consistent with what would be expected for a random sequence.
    """

    def __init__(self, template_length: int = 9, significance_level: float = 0.01):
        """
        Initialize the Overlapping Template Test

        Args:
            template_length: Length m of the template (default is 9)
            significance_level: The significance level (α). Default is 0.01 (1%)
        """
        self.template_length = template_length
        self.significance_level = significance_level
        self.name = "Overlapping Template Test"

        # Test parameters (as defined in NIST paper)
        self.M = 1032  # Length of each substring
        self.K = 5  # Number of degrees of freedom (categories)

        # Default probabilities for K=5 (from NIST implementation)
        self.pi = np.array(
            [0.364091, 0.185659, 0.139381, 0.100571, 0.0704323, 0.139865]
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

    def _compute_probability(self, u: int, eta: float) -> float:
        """
        Compute probability for observing u matches

        Args:
            u: Number of matches
            eta: Expected number of matches (lambda/2)

        Returns:
            float: Probability of observing u matches
        """
        if u == 0:
            return np.exp(-eta)

        # For u > 0, compute sum using logarithms for numerical stability
        log_sum = -np.inf  # Initialize to log(0)
        for l in range(1, u + 1):
            log_term = (
                -eta
                - u * np.log(2)
                + l * np.log(eta)
                - gammaln(l + 1)
                + gammaln(u)
                - gammaln(l)
                - gammaln(u - l + 1)
            )
            log_sum = np.logaddexp(log_sum, log_term)

        return np.exp(log_sum)

    def _count_matches(
        self, sequence: np.ndarray, block_start: int, template: np.ndarray
    ) -> int:
        """
        Count overlapping matches in a block as per NIST SP 800-22 2.8.4
        The window slides over one bit after each examination.

        Args:
            sequence: The full binary sequence
            block_start: Starting position of the current block
            template: Template to match against

        Returns:
            int: Number of matches found
        """
        block = sequence[block_start : block_start + self.M]
        m = len(template)

        # Use numpy's sliding window view for efficiency
        windows = np.lib.stride_tricks.sliding_window_view(block[: self.M - m + 1], m)
        # Compare each window with template
        matches = np.all(windows == template, axis=1)
        # Count total matches (window slides one bit after each match)
        return np.sum(matches)

    def test(
        self,
        binary_data: Union[str, bytes, List[int], np.ndarray],
        template: Union[str, List[int], np.ndarray] = None,
    ) -> dict:
        """
        Run the Overlapping Template Test

        Args:
            binary_data: The sequence to test
            template: Optional specific template to use. If None, uses all ones

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
        N = n // self.M

        if N == 0:
            return {
                "success": False,
                "error": f"Input sequence too short. Length must be at least {self.M}",
                "statistics": {"n": n},
            }

        # Create template (default is all ones)
        if template is None:
            template = np.ones(self.template_length, dtype=int)
        else:
            template = self._convert_to_binary_list(template)
            if len(template) != self.template_length:
                raise ValueError(f"Template length must be {self.template_length}")

        # Calculate test parameters
        lambda_val = (self.M - self.template_length + 1) / (2**self.template_length)
        eta = lambda_val / 2.0

        # Initialize frequency array for counts
        nu = np.zeros(self.K + 1, dtype=int)

        # Process each block
        for i in range(N):
            # Count matches in current block
            W_obs = self._count_matches(sequence, i * self.M, template)

            # Increment appropriate frequency bin
            if W_obs <= self.K - 1:
                nu[W_obs] += 1
            else:
                nu[self.K] += 1

        # Calculate chi-square statistic
        chi_squared = np.sum((nu - N * self.pi) ** 2 / (N * self.pi))

        # Calculate p-value
        p_value = gammaincc(self.K / 2.0, chi_squared / 2.0)

        # Prepare statistics
        stats = {
            "n": n,
            "M": self.M,
            "N": N,
            "template_length": self.template_length,
            "lambda": lambda_val,
            "eta": eta,
            "chi_squared": chi_squared,
            "frequencies": nu.tolist(),
            "template": template.tolist(),
        }

        return {
            "success": bool(p_value >= self.significance_level),
            "p_value": float(p_value),
            "statistics": stats,
        }

    def test_file(self, file_path: Union[str, Path]) -> dict:
        """Run the Overlapping Template Test on a file"""
        with open(file_path, "rb") as f:
            data = f.read()
        return self.test(data)


def format_test_report(test_results: dict) -> str:
    """Format test results as a readable report"""
    if "error" in test_results:
        return (
            "\nOVERLAPPING TEMPLATE OF ALL ONES TEST\n"
            + "-" * 45
            + "\n"
            + f"ERROR: {test_results['error']}\n"
        )

    stats = test_results["statistics"]
    result = "SUCCESS" if test_results["success"] else "FAILURE"

    report = [
        "\n\t\tOVERLAPPING TEMPLATE OF ALL ONES TEST",
        "\t\t-----------------------------------------------",
        "\t\tCOMPUTATIONAL INFORMATION:",
        "\t\t-----------------------------------------------",
        f"\t\t(a) n (sequence_length)      = {stats['n']}",
        f"\t\t(b) m (block length of 1s)   = {stats['template_length']}",
        f"\t\t(c) M (length of substring)  = {stats['M']}",
        f"\t\t(d) N (number of substrings) = {stats['N']}",
        f"\t\t(e) lambda [(M-m+1)/2^m]     = {stats['lambda']:.6f}",
        f"\t\t(f) eta                      = {stats['eta']:.6f}",
        "\t\t-----------------------------------------------",
        "\t\t   F R E Q U E N C Y",
        "\t\t  0   1   2   3   4  ≥5   Chi^2   P-value  Assignment",
        "\t\t-----------------------------------------------",
        f"\t\t{stats['frequencies'][0]:3d} {stats['frequencies'][1]:3d} "
        f"{stats['frequencies'][2]:3d} {stats['frequencies'][3]:3d} "
        f"{stats['frequencies'][4]:3d} {stats['frequencies'][5]:3d}  "
        f"{stats['chi_squared']:f} {test_results['p_value']:.6f} {result}",
    ]

    return "\n".join(report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NIST Overlapping Template Test")
    parser.add_argument("file", type=str, help="Path to the binary file to test")
    parser.add_argument(
        "--template-length",
        type=int,
        default=9,
        help="Length of template (default: 9)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01, help="Significance level (default: 0.01)"
    )

    args = parser.parse_args()

    # Run test
    test = OverlappingTemplateTest(
        template_length=args.template_length, significance_level=args.alpha
    )
    results = test.test_file(args.file)

    # Print report
    print(format_test_report(results))
