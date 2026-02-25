"""
Configuration settings for the Appointment Booking Agent System
"""
from pydantic_settings import BaseSettings
from typing import Optional
from pydantic import ConfigDict  # Ensure ConfigDict is imported


class Settings(BaseSettings):
    model_config = ConfigDict(extra='ignore', env_file=".env", env_file_encoding="utf-8")  # Load from .env file
    # Environment
    ENVIRONMENT: str = "dev"  # dev or prod

    # App
    APP_NAME: str = "Medical Appointment Booking Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # LLM (Ollama + Llama 3.2)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2:latest"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2048

    # External Hospital API — replace with your real API
    EXTERNAL_API_BASE_URL_DEV: str = "https://dev-hmis-api.aarogya.one/api/v1"
    EXTERNAL_API_BASE_URL_PROD: str = "https://beta-hmis.aarogya.one/api/v1"
    EXTERNAL_API_KEY_DEV: Optional[str] = None
    EXTERNAL_API_KEY_PROD: Optional[str] = None
    EXTERNAL_API_TIMEOUT: int = 30

    # Computed properties
    @property
    def EXTERNAL_API_BASE_URL(self) -> str:
        if self.ENVIRONMENT == "prod":
            return self.EXTERNAL_API_BASE_URL_PROD
        return self.EXTERNAL_API_BASE_URL_DEV

    @property
    def EXTERNAL_API_KEY(self) -> Optional[str]:
        if self.ENVIRONMENT == "prod":
            return self.EXTERNAL_API_KEY_PROD
        return self.EXTERNAL_API_KEY_DEV

    # External API Endpoint paths (Aarogya HMIS API)
    DOCTORS_ENDPOINT: str = "/doctors"
    DOCTORS_AVAILABILITY_ENDPOINT: str = "/doctors/availability"
    SLOTS_ENDPOINT: str = "/slots"
    BOOK_APPOINTMENT_ENDPOINT: str = "/appointment/schedule"

    # Default facility ID
    DEFAULT_FACILITY_ID: Optional[str] = None

    # FastAPI
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000


settings = Settings()