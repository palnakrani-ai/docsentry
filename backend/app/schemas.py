"""Pydantic models matching the frozen API contract in ARCHITECTURE.md."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class Citation(BaseModel):
    source: str
    section: str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool
    flags: list[str]
    latencyMs: int


class HealthResponse(BaseModel):
    status: str
    documents: int
    chunks: int


class SourceInfo(BaseModel):
    source: str
    title: str
    sections: int


class SourcesResponse(BaseModel):
    sources: list[SourceInfo]
