import numpy as np
from scipy.special import gammaincc
from pathlib import Path


class RandomExcursionsTest:
    """
    Implementation of NIST's Random Excursions Test

    This test examines the number of cycles having exactly K visits in a cumulative sum
    random walk. The purpose of this test is to detect deviations from the expected
    number of visits to various states in the random walk.
    """

    def __init__(self, significance_level: float = 0.01):
        """
        Initialize the Random Excursions Test

        Args:
            significance_level: The significance level. Default is 0.01 (1%)
        """
        self.significance_level = significance_level
        self.name = 'Random Excursions Test'

        # States to examine
        self.state_x = np.array([-4, -3, -2, -1, 1, 2, 3, 4])

        # Probabilities pi(x) for different states and visit counts (k=0..5)
        # First index is |x|, second index is k (number of visits)
        self.pi = np.array(
            [
                [
                    0.0000000000,
                    0.00000000000,
                    0.00000000000,
                    0.00000000000,
                    0.00000000000,
                    0.0000000000,
                ],
                [
                    0.5000000000,
                    0.25000000000,
                    0.12500000000,
                    0.06250000000,
                    0.03125000000,
                    0.0312500000,
                ],
                [
                    0.7500000000,
                    0.06250000000,
                    0.04687500000,
                    0.03515625000,
                    0.02636718750,
                    0.0791015625,
                ],
                [
                    0.8333333333,
                    0.02777777778,
                    0.02314814815,
                    0.01929012346,
                    0.01607510288,
                    0.0803755143,
                ],
                [
                    0.8750000000,
                    0.01562500000,
                    0.01367187500,
                    0.01196289063,
                    0.01046752930,
                    0.0732727051,
                ],
            ]
        )

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

    def _find_cycles(self, cumsum: np.ndarray) -> np.ndarray:
        """Find indices where cumulative sum returns to zero (cycle endpoints)"""
        zero_crossings = np.where(cumsum == 0)[0]

        if len(zero_crossings) == 0:
            return np.array([len(cumsum)])

        # Add sequence end if it's not already included
        if zero_crossings[-1] != len(cumsum) - 1:
            zero_crossings = np.append(zero_crossings, len(cumsum))

        return zero_crossings

    def test(self, binary_data: str | bytes | list[int] | np.ndarray) -> dict:
        """
        Run the Random Excursions Test

        Args:
            binary_data: The sequence to test

        Returns:
            dict: Test results containing:
                - success: Boolean indicating if test was passed
                - p_values: List of p-values for each state
                - statistics: Additional test statistics
        """
        # Convert input to binary sequence
        sequence = self._convert_to_binary_list(binary_data)
        n = len(sequence)

        # Calculate the partial sums
        S_k = np.zeros(n, dtype=int)
        S_k[0] = 2 * sequence[0] - 1
        for i in range(1, n):
            S_k[i] = S_k[i - 1] + 2 * sequence[i] - 1

        # Find cycles (positions where S_k = 0)
        cycle_ends = self._find_cycles(S_k)
        J = len(cycle_ends)  # Number of cycles

        # Check if we have enough cycles
        constraint = max(0.005 * np.sqrt(n), 500)
        if constraint > J:
            return {
                'success': False,
                'error': 'Insufficient number of cycles',
                'statistics': {'n': n, 'num_cycles': J, 'constraint': constraint},
            }

        # Initialize visit state counts
        nu = np.zeros((6, 8), dtype=int)  # [visit_count, state]

        # Process each cycle
        cycle_start = 0
        for j in range(J):
            cycle_stop = cycle_ends[j]
            # Count visits to each state in the cycle
            cycle = S_k[cycle_start:cycle_stop]
            counter = np.zeros(8, dtype=int)

            # Count visits to states -4 to -1 and 1 to 4
            for state in cycle:
                if -4 <= state <= -1:  # noqa
                    counter[state + 4] += 1
                elif 1 <= state <= 4:  # noqa
                    counter[state + 3] += 1

            # Update nu matrix
            for i in range(8):
                visits = counter[i]
                if visits <= 4:  # noqa
                    nu[visits][i] += 1
                else:
                    nu[5][i] += 1

            cycle_start = cycle_ends[j] + 1

        # Calculate chi-square statistics and p-values for each state
        p_values = []
        chi_squares = []

        for i in range(8):
            x = self.state_x[i]
            x_abs = abs(x)

            # Calculate chi-square statistic
            expected = J * self.pi[x_abs]
            observed = nu[:, i]
            chi_square = np.sum((observed - expected) ** 2 / expected)

            # Calculate p-value
            p_value = gammaincc(2.5, chi_square / 2.0)

            p_values.append(p_value)
            chi_squares.append(chi_square)

        # Prepare statistics
        stats = {
            'n': n,
            'num_cycles': J,
            'chi_squared': chi_squares,
            'states': self.state_x.tolist(),
            'visit_counts': nu.tolist(),
            'constraint': constraint,
        }

        return {
            'success': all(p >= self.significance_level for p in p_values),
            'p_values': float(p_values),
            'statistics': stats,
        }
