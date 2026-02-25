"""
Pydantic models matching Aarogya HMIS API
"""
from pydantic import BaseModel, model_serializer
from typing import Optional, List, Any, Dict
from datetime import datetime


# ── Patient Models ────────────────────────────────────────────────────────────

class BirthDateComponent(BaseModel):
    year: int = 0
    month: int = 0
    day: int = 0


class PermanentAddress(BaseModel):
    pinCode: str = ""
    address: str = ""
    area: str = ""


class PatientDetail(BaseModel):
    permanentAddress: PermanentAddress = PermanentAddress()


class Patient(BaseModel):
    firstName: str
    middleName: Optional[str] = ""
    lastName: str
    mobile: str
    gender: int = 1          # 1=Male, 2=Female
    birthDate: datetime
    birthDateComponent: BirthDateComponent
    healthId: Optional[str] = ""
    healthAddress: Optional[str] = ""
    patientDetail: PatientDetail = PatientDetail()


# ── Appointment Detail ────────────────────────────────────────────────────────

class AppointmentDetail(BaseModel):
    system: int = 1
    consultationType: int = 1
    slotDuration: int = 0
    externalId: Optional[str] = ""           # slot externalId — optional
    healthProfessionalId: Optional[str] = "" # doctor ID — optional
    facilityId: Optional[str] = ""           # facility ID — optional
    chiefComplaints: List[str] = []
    appointentDateTime: datetime             # API typo intentional


# ── Full Booking Request ──────────────────────────────────────────────────────

class AppointmentScheduleRequest(BaseModel):
    patient: Patient
    appointmentDetail: AppointmentDetail


# ── Doctor Availability Response ──────────────────────────────────────────────

class TimeSlot(BaseModel):
    slotId: Optional[str] = None
    externalId: Optional[str] = None
    startTime: Optional[str] = ""
    endTime: Optional[str] = ""
    dateTime: Optional[str] = None
    isAvailable: bool = True


class DoctorWithSlots(BaseModel):
    healthProfessionalId: str
    name: str
    specialization: Optional[str] = None
    facilityId: Optional[str] = None
    facilityName: Optional[str] = None
    qualification: Optional[str] = None
    experience: Optional[str] = None
    availableSlots: List[TimeSlot] = []


# ── Chat Models ───────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    appointment_booked: bool = False
    booking_details: Optional[dict] = None
    timestamp: str


# ── Session Collected Info ────────────────────────────────────────────────────

class CollectedInfo(BaseModel):
    """Tracks all info collected during conversation — all None until filled."""

    # ── Patient details ───────────────────────────────────────────────────────
    firstName:  Optional[str] = None
    middleName: Optional[str] = None
    lastName:   Optional[str] = None
    mobile:     Optional[str] = None
    gender:     Optional[int] = None   # 1=Male, 2=Female
    birthDate:  Optional[str] = None   # YYYY-MM-DD
    age:        Optional[int] = None   # fallback if birthDate unknown

    # Address
    pinCode:  Optional[str] = None
    address:  Optional[str] = None
    area:     Optional[str] = None

    # ── Appointment context ───────────────────────────────────────────────────
    symptoms: Optional[List[str]] = None

    # Doctor selection (set when user picks doctor+slot)
    health_professional_id:  Optional[str] = None
    doctor_name:             Optional[str] = None
    facility_id:             Optional[str] = None
    appointment_date:        Optional[str] = None   # YYYY-MM-DD
    appointment_time:        Optional[str] = None   # HH:MM
    slot_external_id:        Optional[str] = None
    slot_display:            Optional[str] = None   # "10:00 - 10:15" for user display