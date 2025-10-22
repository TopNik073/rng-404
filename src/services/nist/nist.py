import numpy as np
import asyncio
from typing import Final, List, Union
from collections.abc import Mapping, Iterable
from fastapi import HTTPException, UploadFile

from src.services.nist.tests.frequency_test import FrequencyTest
from src.services.nist.tests.block_frequency_test import BlockFrequencyTest
from src.services.nist.tests.runs_test import RunsTest
from src.services.nist.tests.longest_run_ones_test import LongestRunsTest
from src.services.nist.tests.binary_matrix_rank_test import BinaryMatrixRankTest
from src.services.nist.tests.discrete_fourier_transform_test import (
    DiscreteFourierTransformTest,
)
from src.services.nist.tests.non_overlapping_template_test import (
    NonOverlappingTemplateTest,
)
from src.services.nist.tests.overlapping_template_test import OverlappingTemplateTest
from src.services.nist.tests.universal_test import UniversalTest
from src.services.nist.tests.linear_complexity_test import LinearComplexityTest
from src.services.nist.tests.serial_test import SerialTest
from src.services.nist.tests.approximate_entropy_test import ApproximateEntropyTest
from src.services.nist.tests.cumulative_sums_test import CumulativeSumsTest
from src.services.nist.tests.random_excursions_test import RandomExcursionsTest
from src.services.nist.tests.random_excursions_variant_test import (
    RandomExcursionsVariantTest,
)

from src.presentation.api.v1.nist.models import GenerateRequestSchema
from src.core.logger import get_logger

logger = get_logger(__name__)

BLOCK_SIZE: Final[int] = 128


class Nist:
    def __init__(self):
        self.upload_file: UploadFile | None = None
        self.sequence: np.ndarray | List[int] | str | bytes | None = None

        self.tests = {
            "frequency": FrequencyTest(),
            "block_frequency": BlockFrequencyTest(block_size=BLOCK_SIZE),
            "runs": RunsTest(),
            "longest_runs": LongestRunsTest(),
            "matrix_rank": BinaryMatrixRankTest(),
            "dft": DiscreteFourierTransformTest(),
            "template": NonOverlappingTemplateTest(),
            "overlapping_template": OverlappingTemplateTest(),
            "universal": UniversalTest(),
            "linear_complexity": LinearComplexityTest(),
            "serial": SerialTest(),
            "approximate_entropy": ApproximateEntropyTest(),
            "cumulative_sums": CumulativeSumsTest(),
            "random_excursions": RandomExcursionsTest(),
            "random_excursions_variant": RandomExcursionsVariantTest(),
        }

    async def run_test_suite(
        self,
        params: GenerateRequestSchema,
        upload_file: UploadFile | None = None,
    ) -> dict:
        sequence = params.sequence

        included_tests = params.included_tests
        excluded_tests = set(self.tests.keys()) - set(included_tests)
        for test_name in excluded_tests:
            self.tests.pop(test_name)

        if sequence is None and upload_file is None:
            raise HTTPException(400, "No sequence or file uploaded")

        if upload_file is not None:
            file_sequence = await upload_file.read()
            if not file_sequence and not sequence:
                raise HTTPException(400, "No sequence or file uploaded")
            sequence = file_sequence

        results = {}

        for test_name, test in self.tests.items():
            test_results = test.test(sequence)
            if "error" in test_results:
                logger.debug(f"Error: {test_results['error']}")
                test_results["success"] = False

            results[test_name] = test_results
            print(test_name)

        return self._make_json_serializable(results)

    def _make_json_serializable(self, obj):
        """
        Рекурсивно преобразует объект, содержащий numpy-типы, в JSON-сериализуемый формат.
        Поддерживает: скаляры, массивы, списки, словари, вложенные структуры.
        """
        # Скалярные numpy-типы
        if isinstance(
            obj,
            (
                np.integer,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return [self._make_json_serializable(item) for item in obj.tolist()]
        elif isinstance(obj, Mapping):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            # Для всех остальных типов — попытка привести к базовому Python-типу
            try:
                return self._make_json_serializable(np.asarray(obj).item())
            except (ValueError, TypeError, AttributeError):
                # Если не получается — оставляем как есть (может вызвать ошибку при json.dumps)
                return obj


if __name__ == "__main__":
    # # Create example data
    # logger.debug("Generating 1 million random bits...")
    # np.random.seed(42)  # For reproducibility
    # random_bits = np.random.randint(0, 2, size=1000000)
    with open("src/services/nist/test.txt", "r", encoding="utf-8") as f:
        random_bits = np.array([int(bit) for bit in f.read()])

    # Run all tests
    results = asyncio.run(Nist().run_test_suite(random_bits))

    logger.debug("\nDetailed Statistics:")
    logger.debug("=" * 60)

    for test_name, test_results in results.items():
        logger.info(f"\n{test_name.upper()}:")
        logger.info(test_results["success"], end="\n\n")
