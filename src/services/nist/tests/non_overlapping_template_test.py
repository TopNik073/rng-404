import numpy as np
from scipy.special import gammaincc, gammaln
from pathlib import Path
from numba import jit


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
        self.name = 'Non-overlapping Template Test'

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

    def _read_templates(self, templates_dir: str) -> list[np.ndarray]:
        """Read templates from template files"""
        template_file = Path(templates_dir) / f'template{self.template_length}'

        if not Path.exists(Path(template_file)):
            raise FileNotFoundError(f'Template file not found: {template_file}')

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
        """Count non-overlapping matches of template in sequence using JIT optimized version"""
        seq_int32 = sequence.astype(np.int32)
        template_int32 = template.astype(np.int32)
        return _count_matches_jit(seq_int32, template_int32, np.int32(M))

    def test(
        self,
        binary_data: str | bytes | list[int] | np.ndarray,
        templates_dir: str | None = None,
    ) -> dict:
        if templates_dir is None:
            # Use absolute path to templates directory
            templates_dir = Path(__file__).parent / 'templates'
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
                'success': False,
                'error': f'Sequence too short. Need at least {self.B * M} bits.',
                'statistics': {'n': n, 'template_length': self.template_length},
            }

        # Read templates
        try:
            templates = self._read_templates(templates_dir)
        except FileNotFoundError as e:
            return {
                'success': False,
                'error': str(e),
                'statistics': {'n': n, 'template_length': self.template_length},
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
                'success': False,
                'error': f'Expected value (λ={mu}) must be positive',
                'statistics': {'n': n, 'lambda': mu, 'M': M, 'N': N, 'B': self.B},
            }

        # Process templates in chunks for better performance
        results = []
        chunk_size = min(20, len(templates))  # Process 20 templates at a time

        for chunk_start in range(0, len(templates), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(templates))
            chunk_templates = templates[chunk_start:chunk_end]

            # Process each template in the chunk
            for template_idx, template in enumerate(chunk_templates, start=chunk_start):
                seq_int32 = sequence.astype(np.int32)
                template_int32 = template.astype(np.int32)

                W = _process_blocks_jit(seq_int32, template_int32, np.int32(M), np.int32(N))

                # Compute chi-square statistic χ² = Σ((Wᵢ - μ)²/σ²)
                chi_squared = np.sum(((W - mu) ** 2) / sigma2)

                # Calculate p-value using incomplete gamma function
                p_value = gammaincc(N / 2.0, chi_squared / 2.0)  # Store results
                results.append(
                    {
                        'template': template.tolist(),
                        'p_value': float(p_value),
                        'success': bool(p_value >= self.significance_level),
                    }
                )

        # Prepare statistics
        stats = {
            'n': n,
            'M': M,
            'N': N,  # Number of blocks actually used
            'B': self.B,  # Target number of blocks
            'lambda': mu,  # Expected value (λ)
            'variance': sigma2,  # Variance (σ²)
            'template_length': self.template_length,
            'templates_tested': len(templates),
        }

        return {
            'success': all(r['success'] for r in results),
            'results': results,
            'statistics': stats,
        }


@jit(nopython=True, cache=True)
def _count_matches_jit(sequence, template, M):
    """Count non-overlapping matches of template in sequence - JIT optimized version"""
    m = len(template)
    count = 0
    i = 0

    while i <= M - m:
        match = True
        for j in range(m):
            if sequence[i + j] != template[j]:
                match = False
                break

        if match:
            count += 1
            i += m  # Skip ahead by template length for non-overlapping
        else:
            i += 1

    return count

@jit(nopython=True, cache=True)
def _process_blocks_jit(sequence, template, M, N):
    """Process all blocks and count matches - JIT optimized"""
    W = np.zeros(N, dtype=np.float64)
    for i in range(N):
        start_idx = i * M
        end_idx = (i + 1) * M
        block = sequence[start_idx:end_idx]
        W[i] = _count_matches_jit(block, template, M)
    return W