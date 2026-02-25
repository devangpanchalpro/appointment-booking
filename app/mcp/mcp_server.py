"""
MCP Server for Aarogya HMIS
Tools: get_doctors, schedule_appointment
"""
import json
import logging
from datetime import datetime
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from app.api.doctors_cache import doctors_cache
from app.api.external_client import aarogya_api
from app.api.hmis_service import HMISService
from app.config.settings import settings

logger = logging.getLogger(__name__)


# Symptom → Specialization
SYMPTOM_MAP = {
    "chest pain": "cardiology", "heart": "cardiology",
    "joint pain": "orthopedics", "back pain": "orthopedics", "knee pain": "orthopedics",
    "headache": "neurology", "migraine": "neurology",
    "skin": "dermatology", "rash": "dermatology",
    "stomach": "gastroenterology", "nausea": "gastroenterology",
    "fever": "general", "cough": "general", "cold": "general",
}

def _detect_specialization(symptoms: List[str]) -> str:
    combined = " ".join(symptoms).lower()
    for symptom, spec in SYMPTOM_MAP.items():
        if symptom in combined:
            return spec
    return "general medicine"


mcp = FastMCP("AarogyaMCP", "1.0.0")


@mcp.tool()
async def get_doctors_by_symptoms(
    symptoms: List[str],
    facility_id: Optional[str] = None,
) -> str:
    """
    Get doctors with available slots based on patient symptoms.
    Fetches from Aarogya /doctors/availability API.
    
    Args:
        symptoms: Patient symptoms ["fever", "cough"]
        facility_id: Optional facility ID
        
    Returns:
        JSON with doctors list including available time slots
    """
    specialization = _detect_specialization(symptoms)
    logger.info(f"[MCP] get_doctors | symptoms={symptoms} → spec={specialization}")
    
    # Get all doctors from cache
    all_doctors = await doctors_cache.get_doctors(facility_id)
    
    if not all_doctors:
        return json.dumps({
            "success": False,
            "specialization": specialization,
            "doctors": [],
            "message": "No doctors available currently.",
        })
    
    # Filter by specialization
    matched = [
        d for d in all_doctors
        if specialization.lower() in d.get("specialization", "").lower()
    ] or all_doctors  # If no match, show all
    
    # Format for agent
    formatted = []
    for i, doc in enumerate(matched, 1):
        slots = doc.get("availableSlots", [])
        formatted_slots = []
        for j, slot in enumerate(slots[:10], 1):  # Max 10 slots
            formatted_slots.append({
                "index": j,
                "externalId": slot.get("externalId") or slot.get("slotId") or f"slot_{j}",
                "startTime": slot.get("startTime", ""),
                "endTime": slot.get("endTime", ""),
                "dateTime": slot.get("dateTime") or slot.get("startTime", ""),
                "displayTime": f"{slot.get('startTime', '')} - {slot.get('endTime', '')}",
            })
        
        formatted.append({
            "index": i,
            "healthProfessionalId": doc.get("healthProfessionalId", ""),
            "name": doc.get("name", ""),
            "specialization": doc.get("specialization", ""),
            "facilityId": doc.get("facilityId", settings.DEFAULT_FACILITY_ID),
            "qualification": doc.get("qualification", ""),
            "experience": doc.get("experience", ""),
            "availableSlots": formatted_slots,
        })
    
    return json.dumps({
        "success": True,
        "specialization": specialization,
        "count": len(formatted),
        "doctors": formatted,
    })


@mcp.tool()
async def schedule_appointment(
    first_name: str,
    last_name: str,
    mobile: str,
    gender: int,
    birth_date: str,
    symptoms: List[str],
    health_professional_id: str,
    facility_id: str,
    slot_external_id: str,
    slot_datetime: str,
    middle_name: str = "",
    pin_code: str = "",
    address: str = "",
    area: str = "",
) -> str:
    """
    Schedule appointment via /appointment/schedule API.
    
    Args:
        first_name, last_name, mobile, gender, birth_date: Patient info
        symptoms: Chief complaints
        health_professional_id: Doctor ID
        facility_id: Hospital ID
        slot_external_id: Selected slot's externalId
        slot_datetime: ISO datetime string
        
    Returns:
        JSON with booking confirmation
    """
    logger.info(f"[MCP] schedule_appointment | patient={first_name} {middle_name} {last_name}")
    
    try:
        # Parse dates
        bdate = datetime.fromisoformat(birth_date) if "T" in birth_date else datetime.strptime(birth_date, "%Y-%m-%d")
        appt_dt = datetime.fromisoformat(slot_datetime.replace("Z", "+00:00"))
        
        # Build request
        request = HMISService.build_appointment_request(
            first_name=first_name,
            middle_name=middle_name,
            last_name=last_name,
            mobile=mobile,
            gender=gender,
            birth_date=bdate,
            health_professional_id=health_professional_id,
            facility_id=facility_id,
            chief_complaints=symptoms,
            appointment_date_time=appt_dt,
            pin_code=pin_code,
            address=address,
            area=area,
            external_id=slot_external_id,
        )
        
        result = await aarogya_api.schedule_appointment(request)
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"[MCP] schedule error: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e)})


def run_mcp_server():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    run_mcp_server()