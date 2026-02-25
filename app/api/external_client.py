"""
Aarogya HMIS API Client
"""
import httpx
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from app.config.settings import settings
from app.models.schemas import AppointmentScheduleRequest

logger = logging.getLogger(__name__)


class AarogyaAPIClient:
    def __init__(self):
        self.base_url = settings.EXTERNAL_API_BASE_URL
        self.api_key = settings.EXTERNAL_API_KEY
        self.timeout = settings.EXTERNAL_API_TIMEOUT
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-api-key": self.api_key,
        }

    async def get_doctors_availability(
        self,
        facility_id: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        health_professional_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        GET /doctors/availability
        Returns list of doctors with their available time slots.
        """
        if not from_date:
            from_date = datetime.now().strftime("%Y-%m-%d")
        if not to_date:
            to_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        params = {
            "fromDate": from_date,
            "toDate": to_date,
        }
        if health_professional_id:
            params["healthProfessionalId"] = health_professional_id
        if facility_id:
            params["facilityId"] = facility_id
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}{settings.DOCTORS_AVAILABILITY_ENDPOINT}"
                logger.info(f"API: {url} | params: {params}")
                response = await client.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()
                logger.info(f"API Response: {len(data) if isinstance(data, list) else 'dict'} doctors")
                if isinstance(data, dict) and "result" in data:
                    doctors = []
                    for date_entry in data["result"]:
                        for dept in date_entry.get("departments", []):
                            for hp in dept.get("healthProfessionals", []):
                                doctor = {
                                    "healthProfessionalId": hp["healthProfessionalId"],
                                    "healthProfessionalName": hp["healthProfessionalName"],
                                    "department": dept["name"],
                                    "appointmentDate": date_entry["appointmentDate"],
                                    "schedule": hp["schedule"]
                                }
                                doctors.append(doctor)
                    return doctors
                elif isinstance(data, list):
                    return data
                else:
                    return []
        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            return []

    async def schedule_appointment(
        self, booking_request: AppointmentScheduleRequest
    ) -> Dict[str, Any]:
        """
        POST /appointment/schedule
        Books appointment with full patient details + appointment details.
        """
        payload = booking_request.model_dump(mode="json")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}{settings.BOOK_APPOINTMENT_ENDPOINT}"
                logger.info(f"Booking API: {url}")
                logger.debug(f"Payload: {payload}")
                response = await client.post(url, headers=self.headers, json=payload)
                response.raise_for_status()
                result = response.json()
                logger.info(f"Booking Success: {result}")
                return {"success": True, "data": result}
        except httpx.HTTPStatusError as e:
            error = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"Booking failed: {error}")
            return {"success": False, "error": error}
        except Exception as e:
            logger.error(f"Booking error: {str(e)}")
            return {"success": False, "error": str(e)}


# Singleton
aarogya_api = AarogyaAPIClient()