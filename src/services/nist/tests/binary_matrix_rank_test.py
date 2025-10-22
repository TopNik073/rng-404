import numpy as np
from pathlib import Path


class BinaryMatrixRankTest:
    """
    Implementation of NIST's Binary Matrix Rank Test

    This test focuses on the rank of disjoint sub-matrices of the entire sequence.
    The purpose of this test is to check for linear dependence among fixed length
    substrings of the original sequence. The default matrix size is 32x32.
    """

    def __init__(self, matrix_size: int = 32, significance_level: float = 0.01):
        """
        Initialize the Binary Matrix Rank Test

        Args:
            matrix_size: The size of the square matrix (default is 32)
            significance_level: The significance level. Default is 0.01 (1%)
        """
        self.matrix_size = matrix_size
        self.significance_level = significance_level
        self.name = "Binary Matrix Rank Test"

    def _convert_to_binary_list(
        self, input_data: str | bytes | list[int] | np.ndarray
    ) -> np.ndarray:
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
            binary_str = "".join(format(byte, "08b") for byte in input_data)
            return np.array([int(bit) for bit in binary_str])
        if isinstance(input_data, (list, np.ndarray)):
            return np.array(input_data)
        raise ValueError("Unsupported input format")

    def _compute_rank(self, matrix: np.ndarray) -> int:
        """
        Compute the rank of a binary matrix using Gaussian elimination in GF(2)

        Args:
            matrix: Binary matrix as numpy array

        Returns:
            int: Rank of the matrix
        """
        matrix = matrix.copy()  # Make a copy to avoid modifying the original
        n, m = matrix.shape
        rank = 0
        row = 0

        # Process each column
        for col in range(m):
            # Find pivot
            pivot_row = None
            for r in range(row, n):
                if matrix[r, col] == 1:
                    pivot_row = r
                    break

            if pivot_row is not None:
                # Swap rows if needed
                if pivot_row != row:
                    matrix[row], matrix[pivot_row] = (
                        matrix[pivot_row].copy(),
                        matrix[row].copy(),
                    )

                # Eliminate below
                for r in range(row + 1, n):
                    if matrix[r, col] == 1:
                        matrix[r] = (matrix[r] + matrix[row]) % 2

                rank += 1
                row += 1

                if row == n:
                    break

        return rank

    def _compute_probabilities(self) -> tuple[float, float, float]:
        """
        Compute theoretical probabilities for different ranks

        Returns:
            tuple: (p_32, p_31, p_30) probabilities for ranks 32, 31, and less
        """
        M = Q = self.matrix_size

        # Probability for full rank (rank = 32)
        r = M
        product = 1.0
        for i in range(r):
            product *= (
                (1.0 - 2 ** (i - M)) * (1.0 - 2 ** (i - M)) / (1.0 - 2 ** (i - r))
            )
        p_full = 2 ** (r * (M + Q - r) - M * Q) * product

        # Probability for rank = 31
        r = M - 1
        product = 1.0
        for i in range(r):
            product *= (
                (1.0 - 2 ** (i - M)) * (1.0 - 2 ** (i - M)) / (1.0 - 2 ** (i - r))
            )
        p_minus1 = 2 ** (r * (M + Q - r) - M * Q) * product

        # Probability for rank <= 30
        p_rest = 1.0 - (p_full + p_minus1)

        return p_full, p_minus1, p_rest

    def test(self, binary_data: str | bytes | list[int] | np.ndarray) -> dict:
        """
        Run the Binary Matrix Rank Test

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

        # Check if sequence is long enough
        if n < self.matrix_size * self.matrix_size:
            return {
                "success": False,
                "p_value": 0.0,
                "statistics": {
                    "error": f"Insufficient bits to create {self.matrix_size}x{self.matrix_size} matrix",
                    "n": n,
                    "required": self.matrix_size * self.matrix_size,
                },
            }

        # Calculate number of matrices and reshape sequence
        N = n // (self.matrix_size * self.matrix_size)
        discarded_bits = n % (self.matrix_size * self.matrix_size)

        # Reshape sequence into matrices
        matrices = sequence[: N * self.matrix_size * self.matrix_size].reshape(
            N, self.matrix_size, self.matrix_size
        )

        # Calculate theoretical probabilities
        p_full, p_minus1, p_rest = self._compute_probabilities()

        # Count ranks
        F_full = 0  # Count of rank 32
        F_minus1 = 0  # Count of rank 31
        for matrix in matrices:
            rank = self._compute_rank(matrix)
            if rank == self.matrix_size:
                F_full += 1
            elif rank == self.matrix_size - 1:
                F_minus1 += 1

        F_rest = N - (F_full + F_minus1)  # Count of rank ≤ 30

        # Calculate chi-square statistic
        chi_squared = (
            (F_full - N * p_full) ** 2 / (N * p_full)
            + (F_minus1 - N * p_minus1) ** 2 / (N * p_minus1)
            + (F_rest - N * p_rest) ** 2 / (N * p_rest)
        )

        # Calculate p-value
        p_value = np.exp(-chi_squared / 2.0)

        # Ensure p-value is valid
        if p_value < 0 or p_value > 1:
            p_value = 0 if p_value < 0 else 1

        # Prepare statistics
        stats = {
            "n": n,
            "matrix_size": self.matrix_size,
            "num_matrices": N,
            "discarded_bits": discarded_bits,
            "chi_squared": chi_squared,
            "probabilities": {
                f"P_{self.matrix_size}": p_full,
                f"P_{self.matrix_size-1}": p_minus1,
                f"P_{self.matrix_size-2}_or_less": p_rest,
            },
            "frequencies": {
                f"F_{self.matrix_size}": F_full,
                f"F_{self.matrix_size-1}": F_minus1,
                f"F_{self.matrix_size-2}_or_less": F_rest,
            },
        }

        return {
            "success": bool(p_value >= self.significance_level),
            "p_value": float(p_value),
            "statistics": stats,
        }

    def test_file(self, file_path: str | Path) -> dict:
        """
        Run the Binary Matrix Rank Test on a file

        Args:
            file_path: Path to the file containing binary data

        Returns:
            dict: Test results (same as test() method)
        """
        with Path.open(file_path, "rb") as f:
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
    stats = test_results["statistics"]

    if "error" in stats:
        return (
            "\nBINARY MATRIX RANK TEST\n"
            + "-" * 45
            + "\n"
            + f"ERROR: {stats['error']}\n"
            + f"Sequence length: {stats['n']}\n"
            + f"Required length: {stats['required']}\n"
        )

    # Prepare keys for probabilities and frequencies
    m = stats["matrix_size"]
    p_full_key = f"P_{m}"
    p_minus1_key = f"P_{m-1}"
    p_rest_key = f"P_{m-2}_or_less"
    f_full_key = f"F_{m}"
    f_minus1_key = f"F_{m-1}"
    f_rest_key = f"F_{m-2}_or_less"

    report = [
        "\nBINARY MATRIX RANK TEST",
        "-" * 45,
        "COMPUTATIONAL INFORMATION:",
        "-" * 45,
        # Probabilities
        f"(a) Probability P_{m}     = {stats['probabilities'][p_full_key]:.6f}",
        f"(b)           P_{m-1}     = {stats['probabilities'][p_minus1_key]:.6f}",
        f"(c)           P_{m-2}-    = {stats['probabilities'][p_rest_key]:.6f}",
        # Frequencies
        f"(d) Frequency F_{m}     = {stats['frequencies'][f_full_key]:d}",
        f"(e)          F_{m-1}     = {stats['frequencies'][f_minus1_key]:d}",
        f"(f)          F_{m-2}-    = {stats['frequencies'][f_rest_key]:d}",
        f"(g) # of matrices        = {stats['num_matrices']}",
        f"(h) Chi^2               = {stats['chi_squared']:.6f}",
        f"(i) NOTE: {stats['discarded_bits']} BITS WERE DISCARDED.",
        "-" * 45,
        f"{'SUCCESS' if test_results['success'] else 'FAILURE'}",
        f"p_value = {test_results['p_value']:.6f}\n",
    ]

    return "\n".join(report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NIST Binary Matrix Rank Test")
    parser.add_argument("file", type=str, help="Path to the binary file to test")
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=32,
        help="Size of the square matrix (default: 32)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01, help="Significance level (default: 0.01)"
    )

    args = parser.parse_args()

    # Run test
    test = BinaryMatrixRankTest(
        matrix_size=args.matrix_size, significance_level=args.alpha
    )
    results = test.test_file(args.file)

    # Print report
