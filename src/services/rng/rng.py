import datetime
import hashlib
import io
import math
import subprocess
import matplotlib.pyplot as plt
from typing import Final

from fastapi import UploadFile, HTTPException

import numpy as np

from src.integrations.locusonus.client import LocusonusClient
from src.integrations.locusonus.models import LocusonusResponseModel
from src.presentation.api.v1.rng.models import GenerateRequestSchema

SYMBOL_RATE: Final[int] = 48_000
EXC_DURATION: Final[float] = 10.0
LSB_BITS: Final[int] = 8
MIN_BASE: Final[int] = 2
MAX_BASE: Final[int] = 36

class RNG:
    def __init__(self, source_getter: LocusonusClient) -> None:
        self.source_getter = source_getter
        self.uploaded_file: UploadFile | None = None

        self.executed_sources: list[LocusonusResponseModel] = []
        self.executed_images: list[io.BytesIO] = []

        self.seed: bytes | None = None
        self.counter: int = 0

    async def get_random(self, params: GenerateRequestSchema, upload_file: UploadFile | None) -> list[str]:
        self.check_params(params)
        self.uploaded_file = upload_file

        all_entropy: bytes = await self.capture_source()
        self.seed = hashlib.blake2b(all_entropy, digest_size=32).digest()

        return self.generate_from_strings(
            from_str=params.from_num,
            to_str=params.to_num,
            count=params.count,
            base=params.base,
            uniq_only=params.uniq_only,
        )

    def check_params(self, params: GenerateRequestSchema):
        if params.base < MIN_BASE or params.base > MAX_BASE:
            raise HTTPException(400, "base must be between 2 and 36")

        try:
            from_int = int(params.from_num, params.base)
            to_int = int(params.to_num, params.base)
        except ValueError as e:
            raise HTTPException(400, f"Не удалось распарсить from/to в основании {params.base}: {e!s}") from e  # noqa

        if from_int > to_int:
            raise HTTPException(400, "from_num должно быть <= to_num")

        range_size = to_int - from_int + 1
        if params.count > range_size:
            raise HTTPException(400, "count больше размера диапазона (кол-во уникальных значений невозможно)")

    def _bytes_to_int(self, b: bytes) -> int:
        return int.from_bytes(b, "big")

    def _int_to_base(self, value: int, base: int) -> str:
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if value == 0:
            return "0"
        s = ""
        while value > 0:
            value, r = divmod(value, base)
            s = digits[r] + s
        return s

    def _ensure_base_digits(self, needed_digits: int, base: int) -> str:
        """
        Генерирует и возвращает строку цифр в системе base длины >= needed_digits.
        Использует get_bytes(...) под капотом, собирая большие блоки и переводя в base.
        """
        bits_per_digit = math.log2(base)
        bits_needed = math.ceil(bits_per_digit * needed_digits)

        bytes_per_block = max(16, math.ceil(bits_needed / 8))

        base_str_parts: list[str] = []
        generated_digits = 0

        while generated_digits < needed_digits:
            b = self.get_bytes(bytes_per_block)
            v = self._bytes_to_int(b)
            s = self._int_to_base(v, base)

            base_str_parts.append(s)
            generated_digits += len(s)

        return "".join(base_str_parts)

    def generate_from_strings(
            self,
            from_str: str,
            to_str: str,
            count: int,
            base: int,
            uniq_only: bool,
    ) -> list[str]:
        """
        from_str, to_str — числа в виде строк в системе счисления 'base'.
        Возвращает список строк — тех же чисел, представленных в системе base,
        количество = count, все уникальные, принадлежащие диапазону [from, to].
        """
        if count <= 0:
            return []

        from_int = int(from_str, base)
        to_int = int(to_str, base)

        range_size = to_int - from_int + 1

        digits_needed = 1 if to_int == 0 else math.ceil(math.log(to_int + 1, base))

        chunk_digits = max(digits_needed, 4)

        result_list = []
        base_buffer = ""
        read_pos = 0

        def next_candidate() -> int | None:
            nonlocal base_buffer, read_pos
            if read_pos + chunk_digits > len(base_buffer):
                return None
            chunk = base_buffer[read_pos: read_pos + chunk_digits]
            read_pos += chunk_digits
            try:
                val = int(chunk, base)
            except ValueError:
                return None
            return val

        while len(result_list) < count:
            if read_pos + chunk_digits > len(base_buffer):
                want_digits = chunk_digits * 8
                more = self._ensure_base_digits(want_digits, base)
                base_buffer = base_buffer[read_pos:] + more
                read_pos = 0

            cand = next_candidate()
            if cand is None:
                continue

            mapped = from_int + (cand % range_size)

            if mapped < from_int or mapped > to_int:
                mapped = from_int + (mapped - from_int) % range_size

            if not uniq_only:
                result_list.append(mapped)
                continue

            if mapped not in result_list:
                result_list.append(mapped)

        return [self._int_to_base(n, base) for n in result_list]

    def _next_block(self) -> bytes:
        """Генерирует 64 байта новых случайных данных"""
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

        # Make graphic
        self.plot_spectrum(mag)

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

        mask = (1 << bits) - 1
        lsb_vals = uint_view & mask

        if bits == 8:  # noqa
            return lsb_vals.astype(np.uint8).tobytes()
        if bits <= 16:  # noqa
            return lsb_vals.astype(np.uint16).tobytes()
        if bits <= 32:  # noqa
            return lsb_vals.astype(np.uint32).tobytes()
        return lsb_vals.tobytes()  # for 64-bit

    def plot_spectrum(self, mag: np.ndarray):
        """Строит наглядный график спектра и сохраняет его как BytesIO (PNG) в состояние класса"""  # noqa
        energy = np.cumsum(mag)
        energy /= energy[-1]
        cut_index = np.searchsorted(energy, 0.95)
        mag_to_plot = mag[:cut_index]

        plt.figure(figsize=(6, 4))
        plt.plot(mag_to_plot, linewidth=1.2, color='#FD641B')
        plt.title("Спектральный анализ частот")
        plt.xlabel("Сэмплы")
        plt.ylabel("Амплитуда")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120)
        buf.seek(0)
        plt.close()

        self.executed_images.append(buf)