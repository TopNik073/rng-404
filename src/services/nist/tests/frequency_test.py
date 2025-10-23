import numpy as np
from scipy.special import erfc
from scipy.stats import norm
import math
from pathlib import Path


class FrequencyTest:
    """
    Implementation of NIST's Frequency (Monobit) Test
    """

    def __init__(self, significance_level: float = 0.01):
        self.significance_level = significance_level
        self.name = 'Frequency (Monobit) Test'

    def _convert_to_binary_list(self, input_data: str | bytes | list[int] | np.ndarray) -> np.ndarray:
        if isinstance(input_data, str):
            return np.array([int(bit) for bit in input_data])
        if isinstance(input_data, bytes):
            binary_str = ''.join(format(byte, '08b') for byte in input_data)
            return np.array([int(bit) for bit in binary_str])
        if isinstance(input_data, (list, np.ndarray)):
            return np.array(input_data)
        raise ValueError('Unsupported input format')

    def _test_single_block(self, sequence: np.ndarray) -> float:
        """Test a single block (for NIST-style block testing)"""
        n = len(sequence)
        if n == 0:
            return 1.0

        s = 2 * sequence - 1
        s_obs = abs(s.sum()) / math.sqrt(n)
        p_value = erfc(s_obs / math.sqrt(2))
        return p_value

    def test(self, binary_data: str | bytes | list[int] | np.ndarray) -> dict:
        """
        Run the Frequency Test in NIST style (with block partitioning)
        """
        sequence = self._convert_to_binary_list(binary_data)
        n = len(sequence)

        # NIST-style: divide into 100 blocks of 10,000 bits each
        block_size = 10000
        num_blocks = n // block_size

        if num_blocks == 0:
            # For sequences smaller than block_size, use single test
            p_value = self._test_single_block(sequence)
            stats = {
                'n': n,
                'ones_count': np.sum(sequence == 1),
                'zeros_count': np.sum(sequence == 0),
                'ones_proportion': np.mean(sequence),
                'partial_sum': (2 * sequence - 1).sum(),
                's_obs': abs((2 * sequence - 1).sum()) / math.sqrt(n),
                'blocks_tested': 1,
                'proportion_passed': 1.0 if p_value >= self.significance_level else 0.0
            }

            return {
                'success': bool(p_value >= self.significance_level),
                'p_value': float(p_value),
                'statistics': stats,
            }

        # Test each block
        p_values = []
        for i in range(num_blocks):
            block = sequence[i * block_size: (i + 1) * block_size]
            p_val = self._test_single_block(block)
            p_values.append(p_val)

        # NIST proportion test
        pass_count = sum(1 for p in p_values if p >= self.significance_level)
        proportion = pass_count / num_blocks

        # NIST success criteria: proportion >= 0.96
        nist_success = proportion >= 0.96

        # Also compute overall p-value for the entire sequence
        overall_s = 2 * sequence - 1
        overall_s_obs = abs(overall_s.sum()) / math.sqrt(n)
        overall_p_value = erfc(overall_s_obs / math.sqrt(2))

        stats = {
            'n': n,
            'ones_count': np.sum(sequence == 1),
            'zeros_count': np.sum(sequence == 0),
            'ones_proportion': np.mean(sequence),
            'partial_sum': overall_s.sum(),
            's_obs': overall_s_obs,
            'blocks_tested': num_blocks,
            'proportion_passed': proportion,
            'p_values_per_block': [float(p) for p in p_values],
            'nist_style_success': nist_success
        }

        return {
            'success': nist_success,  # Use NIST proportion test result
            'p_value': float(overall_p_value),  # Keep overall p-value for reference
            'statistics': stats,
        }