import logging
import os
from pathlib import Path
import numpy as np

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache

from redis import asyncio

import joblib
from pydantic import BaseModel, ConfigDict, field_validator

logger = logging.getLogger(__name__)
model = None

LOCAL_REDIS_URL = "redis://localhost:6379/0"

@asynccontextmanager
async def lifespan_mechanism(app: FastAPI):
    logging.info("Starting up Lab3 API")

    # Load the Model on Startup
    global model
    model_path = Path(os.path.dirname(os.path.abspath(__file__))).parent / "model_pipeline.pkl"
    print("Model Path",model_path)
    model = joblib.load(model_path) 
    print("Model",model)

    # Load the Redis Cache
    HOST_URL =  os.getenv("REDIS_URL", LOCAL_REDIS_URL)
    redis = asyncio.from_url(HOST_URL, encoding="utf8", decode_responses=True)

    # We initialize the connection to Redis and declare that all keys in the
    # database will be prefixed with w255-cache-predict. Do not change this
    # prefix for the submission.
    FastAPICache.init(RedisBackend(redis), prefix="w255-cache-prediction")

    yield
    # We don't need a shutdown event for our system, but we could put something
    # here after the yield to deal with things during shutdown
    logging.info("Shutting down Lab3 API")


sub_application_housing_predict = FastAPI(lifespan=lifespan_mechanism)

# Use pydantic.Extra.forbid to only except exact field set from client.
# This was not required by the lab.
# Your test should handle the equivalent whenever extra fields are sent.
class House(BaseModel):
    """Data model to parse the request body JSON."""

    model_config = ConfigDict(extra="forbid")

    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

    @field_validator("Latitude")
    @classmethod
    def valid_latitude(cls, v: float) -> float:
        if -90 < v < 90:
            return v
        else:
            raise ValueError("Invalid value for Latitude")

    @field_validator("Longitude")
    @classmethod
    def valid_longitude(cls, v: float) -> float:
        if -180 < v < 180:
            return v
        else:
            raise ValueError("Invalid value for Longitude")

    def to_np(self):
        return np.array(list(vars(self).values())).reshape(1, 8)

class Houses(BaseModel):
    """Data model for a list of House objects."""
    model_config = ConfigDict(extra="forbid")
    houses: list[House]

    def to_np_matrix(self):
        return np.concatenate([house.to_np() for house in self.houses])

class HousePrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction: float 

class HousesPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predictions: list[float]

sub_application_housing_predict = FastAPI(lifespan=lifespan_mechanism)


@sub_application_housing_predict.post("/predict", response_model=HousePrediction)
@cache(expire=300)
async def predict(house: House):
    print("Input 1",house.to_np())
    predict_value = model.predict(house.to_np())
    return_val = HousePrediction(prediction=predict_value[0])
    return return_val

@sub_application_housing_predict.post("/bulk-predict", response_model=HousesPrediction)
@cache(expire=300)
async def multi_predict(houses: Houses):
    np_matrix =  houses.to_np_matrix()
    predict_values = model.predict(np_matrix)
    return_val = HousesPrediction(predictions=predict_values)
    return return_val


# Raises 422 if bad parameter automatically by FastAPI
@sub_application_housing_predict.get("/hello")
async def hello(name: str):
    return {"message": f"Hello {name}"}

@sub_application_housing_predict.get("/health")
async def health():
    return {"time": datetime.now()}


