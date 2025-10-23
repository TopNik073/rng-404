import json
import tempfile
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from io import BytesIO

from fastapi import UploadFile, HTTPException
from fastapi.responses import StreamingResponse, Response

from mutagen.mp3 import MP3

from src.core.config import env_config, app_config
from src.presentation.api.v1.rng.models import GenerateRequestSchema, GeneratorResponseSchema
from src.services.rng.rng import RNG


class RngService:
    def __init__(self, rng: RNG):
        self.rng = rng

    async def generate(
        self, params: GenerateRequestSchema, upload_file: UploadFile | None
    ) -> GeneratorResponseSchema | Response:
        if upload_file:
            file_extension = upload_file.filename.split('.')[-1]
            if file_extension not in app_config.AVAILABLE_AUDIO_FORMAT:
                raise HTTPException(400, 'Invalid file extension, mp3 expected')

        if upload_file and upload_file.filename.endswith('.mp3'):
            with tempfile.NamedTemporaryFile(suffix='.mp3') as tmp:
                tmp.write(await upload_file.read())
                tmp.seek(0)
                audio = MP3(tmp.name)
                duration = audio.info.length

            # We can't proceed big files
            if duration >= env_config.MAX_AUDIO_DURATION:
                raise HTTPException(400, f'Audio duration is too long. Should be less than {env_config.MAX_AUDIO_DURATION} seconds')

            upload_file.file.seek(0)

        random_result: list[str] = await self.rng.get_random(params, upload_file=upload_file)

        msg = MIMEMultipart('mixed')

        # Attach json data
        json_payload = {
            "executed_sources": [src.model_dump() for src in self.rng.executed_sources],
            "seed": self.rng.bytes_to_bits(self.rng.seed),
            "result": random_result,
        }
        json_part = MIMEApplication(json.dumps(json_payload), "json")
        json_part.add_header("Content-Disposition", "attachment", filename="result.json")
        msg.attach(json_part)

        # Attach txt if needed
        if params.format == "txt":
            txt_part = MIMEApplication("\n".join(random_result).encode("utf-8"), "txt")
            txt_part.add_header("Content-Disposition", "attachment", filename="random.txt")
            msg.attach(txt_part)

        # Attach graphics
        for i, img_buf in enumerate(self.rng.executed_images, start=1):
            img_part = MIMEApplication(img_buf.getvalue(), "png")
            img_part.add_header("Content-Disposition", "attachment", filename=f"plot_{i}.png")
            msg.attach(img_part)

        return Response(
            content=msg.as_bytes(),
            media_type=f'multipart/mixed; boundary="{msg.get_boundary()}"',
        )

    async def generate_binary_file(self, length: int) -> StreamingResponse:
        random_result: list[str] = await self.rng.get_random(
            params=GenerateRequestSchema(
                from_num='0',
                to_num='1',
                count=length,
                base=2,
                uniq_only=False,
                format='txt',
            ),
            upload_file=None
        )
        return StreamingResponse(
            content=BytesIO(
                ''.join(random_result).encode('utf-8'),
            ),
            media_type='text/plain',
            headers={'Content-Disposition': 'attachment; filename="random.txt"'},
        )
