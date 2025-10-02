#!/usr/bin/env python3
"""
Super minimal FastAPI app without uvicorn.run
"""

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Super minimal server"}


@app.get("/test")
async def test():
    return {"status": "working"}
