import datetime
import hashlib
import subprocess
from http.client import HTTPException
from typing import Final

from fastapi import UploadFile

import numpy as np

from src.integrations.locusonus.client import LocusonusClient
from src.integrations.locusonus.models import LocusonusResponseModel
from src.presentation.api.v1.rng.models import GenerateRequestSchema

SYMBOL_RATE: Final[int] = 48_000
EXC_DURATION: Final[float] = 1.0
LSB_BITS: Final[int] = 8

class RNG:
    def __init__(self, source_getter: LocusonusClient) -> None:
        self.source_getter = source_getter
        self.uploaded_file: UploadFile | None = None

        self.executed_sources: list[LocusonusResponseModel] = []

        self.seed: bytes | None = None
        self.counter: int = 0

    async def get_random(self, params: GenerateRequestSchema, upload_file: UploadFile) -> str:
        # TODO: implement params bundle
        self.uploaded_file = upload_file

        all_entropy: bytes = await self.capture_source()
        self.seed = hashlib.blake2b(all_entropy, digest_size=32).digest()

        return self.bytes_to_bits(self.get_bytes(125_000))

    def _next_block(self) -> bytes:
        """Генерирует 64 байта новых случайных данных"""
        # BLAKE2b(seed || counter)
        data = self.seed + self.counter.to_bytes(8, "big")
        self.counter += 1
        return hashlib.blake2b(data, digest_size=64).digest()

    def bytes_to_bits(self, data: bytes) -> str:
        return "".join(f"{byte:08b}" for byte in data)

    def get_bytes(self, n: int) -> bytes:
        """Возвращает n случайных байтов"""
        result = bytearray()
        while len(result) < n:
            result.extend(self._next_block())
        return bytes(result[:n])

    def get_start_random_choice(self, len_sources: int) -> list[int]:
        """
        Возвращает три случайных числа для выбора источника звука
        """
        now = datetime.datetime.now()
        microseconds = str(now.microsecond).zfill(6)

        result: list[int] = []
        for i in range(0, 6, 2):
            num = int(microseconds[i:i + 2])
            result.append(num % len_sources)

        return result

    async def capture_source(self) -> bytes:
        all_entropy: bytes = b""
        if self.uploaded_file:
            samples = await self.capture_from_upload(self.uploaded_file)
            feat = self.extract_features(samples)
            all_entropy += feat
        else:
            sources: list[LocusonusResponseModel] = await self.source_getter.get_sources()
            random_source_index = self.get_start_random_choice(len(sources))
            for i in random_source_index:
                source = sources[i]
                self.executed_sources.append(source)
                samples = await self.capture_from_stream(source.url)
                feat = self.extract_features(samples)
                all_entropy += feat

        return all_entropy


    @staticmethod
    async def capture_from_stream(url: str) -> np.ndarray:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            url,
            "-t",
            str(EXC_DURATION),
            "-ar",
            str(SYMBOL_RATE),
            "-ac",
            "1",
            "-f",
            "f32le",
            "pipe:1",
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        raw = p.stdout.read()
        p.wait()
        if len(raw) == 0:
            raise RuntimeError("No data from ffmpeg. Check URL or ffmpeg availability.")
        return np.frombuffer(raw, dtype=np.float32)

    @staticmethod
    async def capture_from_upload(file: UploadFile) -> np.ndarray:
        if not file.content_type.startswith('audio/'):
            raise HTTPException(400, "File must be an audio file")

        try:
            content = await file.read()

            if len(content) == 0:
                raise HTTPException(400, "Uploaded file is empty")

            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-i", "pipe:0",
                "-ar", str(SYMBOL_RATE),
                "-ac", "1",
                "-f", "f32le",
                "pipe:1",
            ]

            p = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            raw, stderr = p.communicate(input=content)

            if p.returncode != 0:
                error_msg = stderr.decode().strip()
                raise HTTPException(500, f"FFmpeg error: {error_msg}")

            if len(raw) == 0:
                raise HTTPException(500, "FFmpeg produced no output")

            return np.frombuffer(raw, dtype=np.float32)

        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(500, f"Error processing audio: {e!s}")  # noqa

    def extract_features(self, samples: np.ndarray) -> bytearray:
        # normalize samples
        if samples.dtype not in (np.float32, np.float64):
            samples = samples.astype("float32") / np.iinfo(samples.dtype).max
        samples = samples - np.mean(samples)
        n = len(samples)

        # window
        w = np.hanning(n)
        x = samples * w

        # FFT
        spec = np.fft.rfft(x)
        mag = np.abs(spec)

        # extract LSB
        lsb_bytes = self.extract_lsb_from_spectrum(mag, bits=LSB_BITS)

        # Build feature bytes
        feat = bytearray()

        # append LSB bytes
        feat += lsb_bytes

        # append sample rate
        salt = hashlib.shake_128(samples.tobytes()).digest(4)
        feat += salt

        return feat

    @staticmethod
    def extract_lsb_from_spectrum(mag, bits=8) -> bytes:
        """
        Извлекает младшие `bits` бит из битового представления
        каждого элемента массива mag (должен быть float32 или float64).

        Возвращает bytes.
        """
        mag = np.asarray(mag)

        if mag.dtype == np.float32:
            uint_view = mag.view(np.uint32)
            max_bits = 32
        elif mag.dtype == np.float64:
            uint_view = mag.view(np.uint64)
            max_bits = 64
        else:
            raise ValueError("mag must be float32 or float64")

        if bits > max_bits or bits <= 0:
            raise ValueError(f"bits must be between 1 and {max_bits}")

        # Маска для младших `bits`
        mask = (1 << bits) - 1
        lsb_vals = uint_view & mask

        if bits == 8:  # noqa
            return lsb_vals.astype(np.uint8).tobytes()
        if bits <= 16:  # noqa
            return lsb_vals.astype(np.uint16).tobytes()
        if bits <= 32:  # noqa
            return lsb_vals.astype(np.uint32).tobytes()
        return lsb_vals.tobytes()  # for 64-bit