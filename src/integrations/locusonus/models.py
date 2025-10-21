from pydantic import BaseModel


class LocusonusResponseModel(BaseModel):
    id: int
    name: str
    url: str
    ext: str
    artist: str
    source: str
    lat: float
    lng: float
    link: str
    city: str
    country: str
