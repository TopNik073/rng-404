class RNG:
    def __init__(self, source_getter):
        self.source_getter = source_getter

    async def get_random(self):
        # TODO: implement RNG here
        return await self.source_getter.get_sources()