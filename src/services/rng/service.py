class RngService:
    def __init__(self, rng):
        self.rng = rng

    async def generate(self):
        return await self.rng.get_random()
