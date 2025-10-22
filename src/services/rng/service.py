import json
import tempfile
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from io import BytesIO

from fastapi import UploadFile, HTTPException
from fastapi.responses import StreamingResponse, Response

from mutagen.mp3 import MP3

from src.core.config import env_config
from src.presentation.api.v1.rng.models import GenerateRequestSchema, GeneratorResponseSchema
from src.services.rng.rng import RNG


class RngService:
    def __init__(self, rng: RNG):
        self.rng = rng

    async def generate(
        self, params: GenerateRequestSchema, upload_file: UploadFile | None
    ) -> GeneratorResponseSchema | StreamingResponse | Response:
        if upload_file and not upload_file.filename.endswith('.mp3'):
            raise HTTPException(400, 'Invalid file extension')

        if upload_file and upload_file.filename.endswith('.mp3'):
            with tempfile.NamedTemporaryFile(suffix='.mp3') as tmp:
                tmp.write(await upload_file.read())
                tmp.seek(0)
                audio = MP3(tmp.name)
                duration = audio.info.length

            if duration >= env_config.MAX_AUDIO_DURATION:
                raise HTTPException(400, 'Audio duration is too long.')

            upload_file.file.seek(0)

        random_result: list[str] = await self.rng.get_random(params, upload_file=upload_file)

        if params.format == 'txt':
            content = '\n'.join(random_result)
            file_like = BytesIO(content.encode('utf-8'))
            return StreamingResponse(
                file_like,
                media_type='text/plain',
                headers={'Content-Disposition': 'attachment; filename="random.txt"'},
            )

        if self.rng.executed_images:
            msg = MIMEMultipart('mixed')

            # JSON часть
            json_part = MIMEApplication(
                json.dumps(
                    {
                        'executed_sources': [source.model_dump() for source in self.rng.executed_sources],
                        'seed': self.rng.bytes_to_bits(self.rng.seed),
                        'result': random_result,
                    }
                ),
                'json',
            )
            json_part.add_header('Content-Disposition', 'attachment', filename='result.json')
            msg.attach(json_part)

            for i, img_buf in enumerate(self.rng.executed_images, start=1):
                img_part = MIMEApplication(img_buf.getvalue(), 'png')
                img_part.add_header('Content-Disposition', 'attachment', filename=f'plot_{i}.png')
                msg.attach(img_part)

            return Response(content=msg.as_bytes(), media_type=f'multipart/mixed; boundary={msg.get_boundary()}')

        return GeneratorResponseSchema(
            executed_sources=self.rng.executed_sources,
            seed=self.rng.bytes_to_bits(self.rng.seed),
            result=random_result,
        )
