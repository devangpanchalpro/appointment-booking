"""
FastAPI Application
"""
import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent.agent import appointment_agent, session_manager
from app.mcp.mcp_client import mcp_client
from app.api.doctors_cache import doctors_cache
from app.config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 {settings.APP_NAME} started")
    logger.info(f"📖 Docs: http://localhost:{settings.FASTAPI_PORT}/docs")
    
    # Pre-fetch doctors on startup
    doctors = await doctors_cache.get_doctors()
    logger.info(f"✅ Cached {len(doctors)} doctors from Aarogya API")
    
    yield
    
    # Shutdown logic here if needed
    logger.info("🛑 Shutting down application")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    appointment_booked: bool = False
    booking_details: Optional[dict] = {}
    timestamp: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "endpoints": {
            "chat": "POST /chat",
            "doctors": "GET /doctors",
            "health": "GET /health",
        }
    }


@app.get("/health")
async def health():
    import httpx
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            ollama_ok = r.status_code == 200
    except:
        pass
    
    tools = await mcp_client.list_tools()
    
    return {
        "status": "healthy",
        "ollama": "running" if ollama_ok else "offline",
        "mcp_tools": len(tools),
        "external_api": settings.EXTERNAL_API_BASE_URL,
    }


@app.get("/doctors", tags=["Doctors"])
async def get_doctors(
    facility_id: Optional[str] = None,
    refresh: bool = False
):
    """
    GET /doctors
    
    Fetch doctors with available slots from Aarogya API.
    Data is cached for 15 minutes unless refresh=true.
    
    Query params:
      - facility_id: Optional facility filter
      - refresh: Force refresh cache (default: false)
    """
    doctors = await doctors_cache.get_doctors(facility_id, force_refresh=refresh)
    return {
        "count": len(doctors),
        "doctors": doctors,
    }


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(req: ChatRequest):
    """
    POST /chat
    
    Chat with the Medical Appointment Booking Agent.
    """
    sid = req.session_id or str(uuid.uuid4())
    
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")
    
    logger.info(f"CHAT | session={sid} | msg={req.message[:80]}")
    
    try:
        result = await appointment_agent.chat(sid, req.message)
        return ChatResponse(
            session_id=sid,
            response=result["response"],
            appointment_booked=result.get("appointment_booked", False),
            booking_details=result.get("booking_details", {}),
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/session/{session_id}", tags=["Session"])
async def get_session(session_id: str):
    s = session_manager.get(session_id)
    return {
        "session_id": session_id,
        "collected_info": s.get("collected"),
        "message_count": len(s.get("messages", [])),
        "booked": s.get("booked", False),
    }


@app.delete("/session/{session_id}", tags=["Session"])
async def reset_session(session_id: str):
    appointment_agent.reset(session_id)
    return {"message": "Session reset", "session_id": session_id}


@app.get("/mcp/tools", tags=["MCP"])
async def list_mcp_tools():
    tools = await mcp_client.list_tools()
    return {"count": len(tools), "tools": tools}