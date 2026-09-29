from typing import Literal
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048, examples=["https://example.com"])


class Contribution(BaseModel):
    feature: str
    value: float
    safe_baseline: float
    risk_contribution: float


class PredictResponse(BaseModel):
    url: str
    classification: Literal["SAFE", "SUSPICIOUS", "PHISHING"]
    risk_score: float
    risk_percentage: int
    reasons: list[str]
    contributions: list[Contribution]
    features: dict[str, float]
    inference: str
