"""
Run the Medical Appointment Booking Agent (FastAPI + MCP + Ollama).
Usage: python run.py
"""
import uvicorn
from app.config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.FASTAPI_HOST,
        port=settings.FASTAPI_PORT,
        reload=settings.DEBUG,
    )
