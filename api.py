"""Optional HTTP service.  Run: uvicorn carbon_audit.api:app --reload"""
from dataclasses import asdict
from fastapi import FastAPI, HTTPException  # type: ignore[import-not-found]
from pydantic import BaseModel  # type: ignore[import-not-found]
from .core import Workload, summarize

app = FastAPI(title="Carbon Audit")


class Describe(BaseModel):
    text: str


@app.post("/audit")
def audit(w: dict, region: str = "montreal"):
    try:
        return summarize(Workload(**w), region=region)
    except TypeError as e:
        raise HTTPException(422, str(e))


@app.post("/parse")
def parse(d: Describe):
    from .llm import extract_workload
    return asdict(extract_workload(d.text))
