"""
Ultra-minimal FastAPI for Render testing
"""
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from Render!"}

@app.get("/api/v1/health")
def health():
    return {"status": "ok"}