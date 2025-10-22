import numpy as np
from scipy.special import gammaincc, gammaln
from pathlib import Path


class NonOverlappingTemplateTest:
    """
    Implementation of NIST's Non-overlapping Template Matching Test

    This test focuses on the number of occurrences of pre-specified target strings.
    The purpose of this test is to detect generators that produce too many occurrences
    of a given non-periodic (aperiodic) pattern.
    """

    def __init__(self, template_length: int = 9, significance_level: float = 0.01):
        """
        Initialize the Non-overlapping Template Test

        Args:
            template_length: Length m of the templates to test (2-21)
            significance_level: The significance level. Default is 0.01 (1%)
        """
        self.template_length = template_length
        self.significance_level = significance_level
        self.name = "Non-overlapping Template Test"

        # Number of templates for each length m
        self.num_templates = {
            2: 2,
            3: 4,
            4: 6,
            5: 12,
            6: 20,
            7: 40,
            8: 74,
            9: 148,
            10: 284,
            11: 568,
            12: 1116,
            13: 2232,
            14: 4424,
            15: 8848,
            16: 17622,
            17: 35244,
            18: 70340,
            19: 140680,
            20: 281076,
            21: 562152,
        }

        # Maximum number of templates to test
        self.MAX_NUM_TEMPLATES = 148

        # Test parameters (as per NIST SP 800-22)
        self.B = 8  # Number of blocks (N in NIST paper)
        self.K = 5  # Maximum number of matches to track
        # M (block length) will be calculated based on sequence length
        # Recommended M ≈ 1032 for million-bit sequence with m = 9

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

    def _read_templates(self, templates_dir: str) -> list[np.ndarray]:
        """Read templates from template files"""
        template_file = Path(templates_dir) / f"template{self.template_length}"

        if not Path.exists(Path(template_file)):
            raise FileNotFoundError(f"Template file not found: {template_file}")

        # Read all lines and process them
        templates = []
        current_template = []

        with Path.open(Path(template_file)) as f:
            for line in f:
                if not line.strip():  # Skip empty lines
                    continue

                try:
                    # Try to get space-separated bits first
                    bits = line.split()
                    if len(bits) > 1:  # Line contains multiple bits
                        template = [int(bit) for bit in bits]
                        if len(template) == self.template_length:
                            templates.append(np.array(template))
                    else:  # Line contains a single bit
                        bit = int(line)
                        current_template.append(bit)
                        if len(current_template) == self.template_length:
                            templates.append(np.array(current_template))
                            current_template = []
                except ValueError:
                    # Skip invalid lines
                    current_template = []

        # Check if we have an incomplete template at the end
        if current_template and len(current_template) == self.template_length:
            templates.append(np.array(current_template))

        return templates[: self.MAX_NUM_TEMPLATES]

    def _compute_probabilities(self, lambda_val: float) -> np.ndarray:
        """Compute probabilities for template matches"""
        pi = np.zeros(self.K + 1)

        # Compute first two probabilities
        sum_prob = 0
        for i in range(2):
            pi[i] = np.exp(-lambda_val + i * np.log(lambda_val) - gammaln(i + 1))
            sum_prob += pi[i]

        # Store sum in first position
        pi[0] = sum_prob

        # Compute remaining probabilities
        for i in range(2, self.K + 1):
            pi[i - 1] = np.exp(-lambda_val + i * np.log(lambda_val) - gammaln(i + 1))
            sum_prob += pi[i - 1]

        # Last probability is complement
        pi[self.K] = 1 - sum_prob

        return pi

    def _count_matches(self, sequence: np.ndarray, template: np.ndarray, M: int) -> int:
        """Count non-overlapping matches of template in sequence using vectorized operations"""
        m = len(template)
        # Create a view of all possible template-sized windows
        windows = np.lib.stride_tricks.sliding_window_view(sequence[:M], m)
        # Find matches using vectorized comparison
        matches = np.all(windows == template, axis=1)
        # Find indices of matches
        match_indices = np.nonzero(matches)[0]

        if len(match_indices) == 0:
            return 0

        # Count non-overlapping matches
        count = 1
        last_idx = match_indices[0]

        for idx in match_indices[1:]:
            if idx >= last_idx + m:
                count += 1
                last_idx = idx

        return count

    def test(
        self,
        binary_data: str | bytes | list[int] | np.ndarray,
        templates_dir: str | None = None,
    ) -> dict:
        if templates_dir is None:
            # Use absolute path to templates directory
            templates_dir = Path(__file__).parent / "templates"
        """
        Run the Non-overlapping Template Test

        Args:
            binary_data: The sequence to test
            templates_dir: Directory containing template files

        Returns:
            dict: A dictionary containing:
                - 'success': Boolean indicating if all tests were passed
                - 'results': List of individual template test results
                - 'statistics': Additional test statistics
        """
        # Convert input to binary sequence
        sequence = self._convert_to_binary_list(binary_data)
        n = len(sequence)

        # Compute block size M as per NIST SP 800-22
        if n >= 1000000:  # noqa
            M = 1032  # Recommended size for million-bit sequence
        else:
            M = n // self.B  # For smaller sequences

        # Adjust number of blocks based on sequence length
        N = n // M  # This is B in NIST paper
        if N < self.B:
            return {
                "success": False,
                "error": f"Sequence too short. Need at least {self.B * M} bits.",
                "statistics": {"n": n, "template_length": self.template_length},
            }

        # Read templates
        try:
            templates = self._read_templates(templates_dir)
        except FileNotFoundError as e:
            return {
                "success": False,
                "error": str(e),
                "statistics": {"n": n, "template_length": self.template_length},
            }

        # Compute test parameters
        # Compute mean (λ) and variance (σ²) as per NIST SP 800-22
        mu = (M - self.template_length + 1) / (2.0**self.template_length)
        sigma2 = M * (
            1.0 / (2.0**self.template_length)
            - (2.0 * self.template_length - 1.0) / (2.0 ** (2.0 * self.template_length))
        )

        if mu <= 0:
            return {
                "success": False,
                "error": f"Expected value (λ={mu}) must be positive",
                "statistics": {"n": n, "lambda": mu, "M": M, "N": N, "B": self.B},
            }

        # Process templates in chunks for better performance
        results = []
        chunk_size = min(20, len(templates))  # Process 20 templates at a time

        for chunk_start in range(0, len(templates), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(templates))
            chunk_templates = templates[chunk_start:chunk_end]

            # Process each template in the chunk
            for template_idx, template in enumerate(chunk_templates, start=chunk_start):
                # Process each block
                W = np.zeros(N)
                blocks = [sequence[i * M : (i + 1) * M] for i in range(N)]

                # Process blocks in parallel using numpy operations
                for i, block in enumerate(blocks):
                    W[i] = self._count_matches(block, template, M)

                # Compute chi-square statistic χ² = Σ((Wᵢ - μ)²/σ²)
                chi_squared = np.sum(((W - mu) ** 2) / sigma2)

                # Calculate p-value using incomplete gamma function
                p_value = gammaincc(N / 2.0, chi_squared / 2.0)  # Store results
                results.append(
                    {
                        "template": template.tolist(),
                        "template_idx": template_idx,
                        "W": W.tolist(),
                        "chi_squared": chi_squared,
                        "p_value": float(p_value),
                        "success": bool(p_value >= self.significance_level),
                    }
                )

        # Prepare statistics
        stats = {
            "n": n,
            "M": M,
            "N": N,  # Number of blocks actually used
            "B": self.B,  # Target number of blocks
            "lambda": mu,  # Expected value (λ)
            "variance": sigma2,  # Variance (σ²)
            "template_length": self.template_length,
            "templates_tested": len(templates),
        }

        return {
            "success": all(r["success"] for r in results),
            "results": results,
            "statistics": stats,
        }

    def test_file(self, file_path: str | Path, templates_dir: str | None = None) -> dict:
        """Run the Non-overlapping Template Test on a file"""
        with Path.open(file_path, "rb") as f:
            data = f.read()
        return self.test(data, templates_dir)


def format_test_report(test_results: dict) -> str:
    """Format test results as a readable report"""
    if "error" in test_results:
        return (
            "\nNON-OVERLAPPING TEMPLATE TEST\n"
            + "-" * 45
            + "\n"
            + f"ERROR: {test_results['error']}\n"
        )

    stats = test_results["statistics"]

    report = [
        "\nNON-OVERLAPPING TEMPLATE TEST",
        "-" * 75,
        "COMPUTATIONAL INFORMATION:",
        "-" * 75,
        f"LAMBDA = {stats['lambda']:.6f}",
        f"M (block length) = {stats['M']}",
        f"N (number of blocks) = {stats['N']}",
        f"m (template length) = {stats['template_length']}",
        f"n (sequence length) = {stats['n']}",
        "-" * 75,
        "                F R E Q U E N C Y",
        "Template  W_1  W_2  W_3  W_4  W_5  W_6  W_7  W_8    Chi^2   P-value Result  Index",
    ]

    for result in test_results["results"]:
        template_str = "".join(map(str, result["template"]))
        W_str = " ".join(f"{w:3d}" for w in result["W"])
        status = "SUCCESS" if result["success"] else "FAILURE"

        report.append(
            f"{template_str} {W_str}  {result['chi_squared']:7.4f} {result['p_value']:.6f} {status:7} {result['template_idx']:4d}"
        )

    return "\n".join(report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NIST Non-overlapping Template Test")
    parser.add_argument("file", type=str, help="Path to the binary file to test")
    parser.add_argument(
        "--template-length",
        type=int,
        default=9,
        help="Length of templates to test (2-21, default: 9)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01, help="Significance level (default: 0.01)"
    )
    parser.add_argument(
        "--templates-dir",
        type=str,
        default="templates",
        help="Directory containing template files",
    )

    args = parser.parse_args()

    # Run test
    test = NonOverlappingTemplateTest(
        template_length=args.template_length, significance_level=args.alpha
    )
    results = test.test_file(args.file, args.templates_dir)

    # Print report
