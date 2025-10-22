from collections.abc import Mapping
from typing import Final

import numpy as np
from fastapi import UploadFile, HTTPException

from src.presentation.api.v1.nist.models import NistRequestSchema
from src.services.nist.tests.approximate_entropy_test import ApproximateEntropyTest
from src.services.nist.tests.binary_matrix_rank_test import BinaryMatrixRankTest
from src.services.nist.tests.block_frequency_test import BlockFrequencyTest
from src.services.nist.tests.cumulative_sums_test import CumulativeSumsTest
from src.services.nist.tests.discrete_fourier_transform_test import DiscreteFourierTransformTest
from src.services.nist.tests.frequency_test import FrequencyTest
from src.services.nist.tests.linear_complexity_test import LinearComplexityTest
from src.services.nist.tests.longest_run_ones_test import LongestRunsTest
from src.services.nist.tests.non_overlapping_template_test import NonOverlappingTemplateTest
from src.services.nist.tests.overlapping_template_test import OverlappingTemplateTest
from src.services.nist.tests.random_excursions_test import RandomExcursionsTest
from src.services.nist.tests.random_excursions_variant_test import RandomExcursionsVariantTest
from src.services.nist.tests.runs_test import RunsTest
from src.services.nist.tests.serial_test import SerialTest
from src.services.nist.tests.universal_test import UniversalTest

from src.core.logger import get_logger

logger = get_logger(__name__)


BLOCK_SIZE: Final[int] = 128


class NistService:
    def __init__(self) -> None:
        self.upload_file: UploadFile | None = None
        self.sequence: list[int] | str | bytes | None = None

        self.tests = {
            'frequency': FrequencyTest(),
            'block_frequency': BlockFrequencyTest(block_size=BLOCK_SIZE),
            'runs': RunsTest(),
            'longest_runs': LongestRunsTest(),
            'matrix_rank': BinaryMatrixRankTest(),
            'dft': DiscreteFourierTransformTest(),
            'template': NonOverlappingTemplateTest(),
            'overlapping_template': OverlappingTemplateTest(),
            'universal': UniversalTest(),
            'linear_complexity': LinearComplexityTest(),
            'serial': SerialTest(),
            'approximate_entropy': ApproximateEntropyTest(),
            'cumulative_sums': CumulativeSumsTest(),
            'random_excursions': RandomExcursionsTest(),
            'random_excursions_variant': RandomExcursionsVariantTest(),
        }

    async def check(self, params: NistRequestSchema, upload_file: UploadFile | None) -> dict:
        sequence = params.sequence

        included_tests = params.included_tests
        excluded_tests = set(self.tests.keys()) - set(included_tests)
        for test_name in excluded_tests:
            self.tests.pop(test_name)

        if sequence is None and upload_file is None:
            raise HTTPException(400, 'No sequence or file uploaded')

        if upload_file is not None:
            file_sequence = await upload_file.read()
            if not file_sequence and not sequence:
                raise HTTPException(400, 'No sequence or file uploaded')
            sequence = file_sequence

        results = {}

        for test_name, test in self.tests.items():
            test_results = test.test(sequence)
            if 'error' in test_results:
                logger.debug(f'Error: {test_results["error"]}')
                test_results['success'] = False

            results[test_name] = test_results

        return self._make_json_serializable(results)

    def _make_json_serializable(self, obj):  # noqa
        """
        Рекурсивно преобразует объект, содержащий numpy-типы, в JSON-сериализуемый формат.
        Поддерживает: скаляры, массивы, списки, словари, вложенные структуры.
        """
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
        if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return [self._make_json_serializable(item) for item in obj.tolist()]
        if isinstance(obj, Mapping):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        # Для всех остальных типов — попытка привести к базовому Python-типу
        try:
            return self._make_json_serializable(np.asarray(obj).item())
        except (ValueError, TypeError, AttributeError):
            # Если не получается — оставляем как есть (может вызвать ошибку при json.dumps)
            return obj
