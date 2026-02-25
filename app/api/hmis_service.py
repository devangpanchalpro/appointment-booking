"""
HMIS API Service — Fully dynamic, no static patient data
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.api.external_client import aarogya_api
from app.models.schemas import (
    AppointmentScheduleRequest,
    Patient,
    AppointmentDetail,
    BirthDateComponent,
    PermanentAddress,
    PatientDetail,
)

logger = logging.getLogger(__name__)


class HMISService:
    """All data is dynamic — collected from conversation."""

    @staticmethod
    def build_appointment_request(
        first_name: str,
        middle_name: str,
        last_name: str,
        mobile: str,
        gender: int,
        birth_date: datetime,
        health_professional_id: str,
        facility_id: str,
        chief_complaints: List[str],
        appointment_date_time: datetime,
        pin_code: str = "",
        address: str = "",
        area: str = "",
        external_id: Optional[str] = None,
    ) -> AppointmentScheduleRequest:
        """Build AppointmentScheduleRequest from patient and appointment data."""
        bd_comp = BirthDateComponent(
            year=birth_date.year,
            month=birth_date.month,
            day=birth_date.day,
        )

        patient = Patient(
            firstName  = first_name,
            middleName = middle_name,
            lastName   = last_name,
            mobile     = mobile,
            gender     = gender,
            birthDate  = birth_date,
            birthDateComponent = bd_comp,
            healthId      = "",
            healthAddress = "",
            patientDetail = PatientDetail(
                permanentAddress=PermanentAddress(
                    pinCode = pin_code,
                    address = address,
                    area    = area,
                )
            ),
        )

        appointment_detail = AppointmentDetail(
            system               = 1,
            consultationType     = 1,
            slotDuration         = 0,
            externalId           = external_id or "",
            healthProfessionalId = health_professional_id,
            facilityId           = facility_id,
            chiefComplaints      = chief_complaints,
            appointentDateTime   = appointment_date_time,
        )

        return AppointmentScheduleRequest(patient=patient, appointmentDetail=appointment_detail)

    async def get_doctors_availability(
        self,
        facility_id: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        health_professional_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await aarogya_api.get_doctors_availability(
            facility_id=facility_id,
            from_date=from_date,
            to_date=to_date,
            health_professional_id=health_professional_id,
        )

    async def schedule_appointment(
        self,
        first_name: str,
        last_name: str,
        mobile: str,
        gender: int,
        birth_date: datetime,
        health_professional_id: str,
        facility_id: str,
        chief_complaints: List[str],
        appointment_date_time: datetime,
        middle_name: str = "",
        pin_code: str = "",
        address: str = "",
        area: str = "",
        external_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        request = self.build_appointment_request(
            first_name=first_name,
            middle_name=middle_name,
            last_name=last_name,
            mobile=mobile,
            gender=gender,
            birth_date=birth_date,
            health_professional_id=health_professional_id,
            facility_id=facility_id,
            chief_complaints=chief_complaints,
            appointment_date_time=appointment_date_time,
            pin_code=pin_code,
            address=address,
            area=area,
            external_id=external_id,
        )
        logger.info(f"Scheduling for {first_name} {last_name} with Dr. {health_professional_id}")
        return await aarogya_api.schedule_appointment(request)


hmis_service = HMISService()