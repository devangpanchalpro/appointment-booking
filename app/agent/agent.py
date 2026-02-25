"""
Medical Appointment Booking Agent
LLM-powered info extraction + Dynamic booking via Aarogya HMIS API
"""
import json
import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import httpx

from app.api.doctors_cache import doctors_cache
from app.api.external_client import aarogya_api
from app.models.schemas import (
    AppointmentScheduleRequest,
    Patient,
    AppointmentDetail,
    BirthDateComponent,
    PermanentAddress,
    PatientDetail,
    CollectedInfo,
)
from app.config.settings import settings

logger = logging.getLogger(__name__)


# ── Symptom → Department ──────────────────────────────────────────────────────

SYMPTOM_DEPARTMENT_MAP = {
    "chest pain": ["Cardiology"],
    "heart":      ["Cardiology"],
    "headache":   ["Neurology"],
    "migraine":   ["Neurology"],
    "fever":      ["Medicine", "General"],
    "cough":      ["Medicine", "General"],
    "cold":       ["Medicine", "General"],
    "pain":       ["Medicine", "Orthopaedics"],
    "nausea":     ["Medicine", "Gastroenterology"],
    "vomiting":   ["Medicine", "Gastroenterology"],
    "skin":       ["Dermatology"],
    "rash":       ["Dermatology"],
}


def filter_doctors_by_symptoms(doctors: List[Dict], symptoms: List[str]) -> List[Dict]:
    if not symptoms:
        return doctors
    relevant_depts = set()
    for symptom in symptoms:
        for key, depts in SYMPTOM_DEPARTMENT_MAP.items():
            if key in symptom.lower():
                relevant_depts.update([d.lower() for d in depts])
    if not relevant_depts:
        return doctors
    filtered = [
        d for d in doctors
        if any(dep in d.get("department", "").lower() for dep in relevant_depts)
    ]
    return filtered or doctors


# ── Doctor display ────────────────────────────────────────────────────────────

def format_doctors_for_display(doctors: List[Dict]) -> str:
    if not doctors:
        return "No doctors available right now."
    
    from datetime import datetime
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    current_time_str = now.strftime("%H:%M")
    
    lines = []
    for i, doc in enumerate(doctors, 1):
        name  = doc.get("healthProfessionalName", "Unknown")
        dept  = doc.get("department", "")
        date  = doc.get("appointmentDate", "")
        lines.append(f"\n{i}. Dr. {name} ({dept}) — {date}")
        slot_num = 1
        for sched in doc.get("schedule", []):
            session_name   = sched.get("session", "")
            available_slots = [s for s in sched.get("slots", []) if s.get("isAvailable", False)]
            
            # Filter out past slots if appointment date is today
            if date == today_str:
                available_slots = [
                    s for s in available_slots 
                    if s.get("from", "") > current_time_str
                ]
            
            if available_slots:
                lines.append(f"   [{session_name}]")
                for slot in available_slots[:5]:
                    lines.append(f"     Slot {slot_num}: {slot.get('from','')} - {slot.get('to','')}")
                    slot_num += 1
    lines.append("\nReply with: doctor number + slot number  (e.g. '1 2' or 'doctor 1 slot 2')")
    return "\n".join(lines)


# ── Session Manager ───────────────────────────────────────────────────────────

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Dict] = {}

    def get(self, sid: str) -> Dict:
        if sid not in self._sessions:
            self._sessions[sid] = {
                "messages": [],
                "collected": CollectedInfo().model_dump(),
                "doctors":   [],
                "stage":     "start",   # start → doctors_shown → selected → booking
                "booked":    False,
            }
        return self._sessions[sid]

    def add_message(self, sid: str, role: str, content: str):
        self.get(sid)["messages"].append({"role": role, "content": content})

    def messages(self, sid: str) -> List[Dict]:
        return self.get(sid)["messages"]

    def update_collected(self, sid: str, updates: Dict):
        collected = self.get(sid)["collected"]
        def snake_to_camel(s: str) -> str:
            parts = s.split("_")
            return parts[0] + ''.join(p.title() for p in parts[1:]) if len(parts) > 1 else s

        def camel_to_snake(s: str) -> str:
            return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()

        for k, v in updates.items():
            # Ignore empty updates
            if v is None or v == "" or v == []:
                continue

            # Always write the provided key if present in collected, else still write it to collected
            collected[k] = v

            # Also write normalized variants so both naming styles exist
            if "_" in k:
                camel = snake_to_camel(k)
                collected[camel] = v
            else:
                snake = camel_to_snake(k)
                collected[snake] = v

    def collected(self, sid: str) -> Dict:
        return self.get(sid)["collected"]

    def reset(self, sid: str):
        self._sessions.pop(sid, None)


session_manager = SessionManager()


# ── Doctor/Slot selection detector ───────────────────────────────────────────

def detect_doctor_slot_selection(message: str, doctors: List[Dict]) -> Optional[Dict]:
    """
    Parses messages like:
      "doctor 1 slot 2", "1 2", "dr 2 slot 1", "choose doctor 1 and slot 3"
    Returns {"doctor": doc_dict, "slot": slot_dict} or None.
    """
    if not doctors:
        return None

    msg_lower = message.lower()
    numbers = re.findall(r'\b(\d+)\b', message)

    # Two bare numbers → first = doctor index, second = slot index
    if len(numbers) >= 2:
        doc_idx  = int(numbers[0]) - 1
        slot_idx = int(numbers[1]) - 1
        if 0 <= doc_idx < len(doctors):
            doc = doctors[doc_idx]
            all_slots = _flatten_available_slots(doc, filter_today_future=True)
            if 0 <= slot_idx < len(all_slots):
                return {"doctor": doc, "slot": all_slots[slot_idx]}

    # Doctor by name
    for doc in doctors:
        name = doc.get("healthProfessionalName", "").lower()
        if name and name in msg_lower:
            slot_match = re.search(r'slot\s*(\d+)', msg_lower)
            if slot_match:
                slot_idx  = int(slot_match.group(1)) - 1
                all_slots = _flatten_available_slots(doc, filter_today_future=True)
                if 0 <= slot_idx < len(all_slots):
                    return {"doctor": doc, "slot": all_slots[slot_idx]}

    return None


def _flatten_available_slots(doc: Dict, filter_today_future: bool = False) -> List[Dict]:
    from datetime import datetime
    slots = []
    for sched in doc.get("schedule", []):
        for s in sched.get("slots", []):
            if s.get("isAvailable", False):
                slots.append(s)
    
    # Filter out past slots if appointment date is today and filter_today_future is True
    if filter_today_future:
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        current_time_str = now.strftime("%H:%M")
        
        appointment_date = doc.get("appointmentDate", "")
        if appointment_date == today_str:
            slots = [s for s in slots if s.get("from", "") > current_time_str]
    
    return slots


def find_doctor_by_name(doctor_name: str, doctors: List[Dict]) -> Optional[Dict]:
    """
    Find doctor by name (fuzzy matching).
    Handles "Dr. Sharma", "Sharma", "sharma", etc.
    """
    if not doctor_name or not doctors:
        return None
    
    name_lower = doctor_name.lower().replace("dr.", "").strip()
    
    for doc in doctors:
        doc_name = doc.get("healthProfessionalName", "").lower()
        # Check if the extracted name is part of doctor's name
        if name_lower in doc_name or doc_name in name_lower:
            return doc
    
    return None


def select_slot_by_time(doctor: Dict, appointment_slot: str) -> Optional[Dict]:
    """
    Select a slot matching the appointment_slot string.
    Handles "10am", "2:30pm", "morning", "afternoon", "evening", "tomorrow 5pm", etc.
    """
    if not doctor or not appointment_slot:
        return None
    
    slot_lower = appointment_slot.lower().strip()
    all_slots = _flatten_available_slots(doctor, filter_today_future=True)
    
    if not all_slots:
        return None
    
    # Try exact time match (10:00, 10am, 2:30pm, etc.)
    time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', slot_lower)
    if time_match:
        hh = int(time_match.group(1))
        mm = int(time_match.group(2) or 0)
        ampm = time_match.group(3)
        if ampm == 'pm' and hh != 12:
            hh += 12
        if ampm == 'am' and hh == 12:
            hh = 0
        time_str = f"{hh:02d}:{mm:02d}"
        
        for s in all_slots:
            s_from = (s.get('from') or s.get('startTime') or '').strip().lower()
            if time_str in s_from:
                return s
    
    # Try time preferences (morning, afternoon, evening)
    prefer = None
    if 'morning' in slot_lower:
        prefer = 'morning'
    elif 'afternoon' in slot_lower:
        prefer = 'afternoon'
    elif 'evening' in slot_lower:
        prefer = 'evening'
    
    if prefer:
        for s in all_slots:
            s_from = (s.get('from') or s.get('startTime') or '').strip().lower()
            try:
                h = int(s_from.split(':')[0])
            except:
                h = None
            
            if prefer == 'morning' and h is not None and 6 <= h < 12:
                return s
            elif prefer == 'afternoon' and h is not None and 12 <= h < 17:
                return s
            elif prefer == 'evening' and h is not None and 17 <= h < 22:
                return s
    
    # Fallback to first available slot
    return all_slots[0] if all_slots else None


# ── Context summary ───────────────────────────────────────────────────────────

def build_context_summary(collected: Dict) -> str:
    filled = {k: v for k, v in collected.items() if v is not None and v != "" and v != []}
    if not filled:
        return "Nothing collected yet."
    return "\n".join(f"  {k}: {v}" for k, v in filled.items())


# ── Missing field checker ─────────────────────────────────────────────────────

REQUIRED_PATIENT_FIELDS = [
    "firstName", "lastName", "mobile", "gender", "pinCode", "address", "area"
]


def missing_patient_fields(collected: Dict) -> List[str]:
    missing = [k for k in REQUIRED_PATIENT_FIELDS if not collected.get(k)]
    # Need either birthDate OR age
    if not collected.get("birthDate") and not collected.get("age"):
        missing.append("dateOfBirth (DD/MM/YYYY) or age")
    return missing


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Medical Appointment Booking Agent for Aarogya hospital.
Help patients book appointments through natural, friendly conversation.

## FLOW:
1. User mentions symptoms → you show available doctors with slots
2. User picks doctor + slot (e.g. "doctor 1 slot 2")
3. You ask for all patient details in ONE go:
   "Please share: full name (first middle last), mobile, gender, date of birth or age, full address, pincode, area"
4. User replies naturally — you extract all info from their message
5. When all info collected → confirm and book automatically

## CURRENT SESSION STATE:
{context}

## MISSING FIELDS (still needed before booking):
{missing}

## RULES:
- Do NOT re-ask fields already in session state above
- If missing fields is empty AND doctor is selected → booking will happen automatically
- Emergency (severe chest pain, unconscious) → tell user to call 108 immediately
- Do NOT diagnose — only help book appointments
- Reply in same language as user (Gujarati / Hindi / English)
"""

EXTRACTION_PROMPT = """You are a strict JSON extraction assistant.

Extract information from the text below.
Return ONLY a valid JSON object. No explanation, no markdown, no extra text.

Fields to extract:
{
  "firstName":  string or null,
  "middleName": string or null,
  "lastName":   string or null,
  "mobile":     "10-digit string" or null,
  "gender":     "Male" or "Female" or null,
  "birthDate":  "YYYY-MM-DD" or null,
  "age":        integer or null,
  "pinCode":    "6-digit string" or null,
  "address":    string or null,
  "area":       string or null,
  "symptoms":   ["symptom1", "symptom2"] or null,
  "doctor_name": string (with "Dr." if mentioned) or null,
  "appointment_slot": string (e.g., "10am", "14:30", "tomorrow 5pm") or null
}

Rules:
- Name: Parse full name carefully. For Indian names, typically: firstName (given name), middleName (father's name), lastName (surname/family name)
  - If 1 word: firstName only
  - If 2 words: firstName + lastName
  - If 3 words: firstName + middleName + lastName
  - If 4+ words: firstName + middleName + combine rest as lastName
  - Example: "panchal devang hasmukhbhai" → firstName="panchal", middleName="devang", lastName="hasmukhbhai"
  - Do NOT duplicate any part of the name
- mobile: exactly 10 digits (Indian number), no spaces or dashes
- gender: "Male" or "Female" or null. Map variants: male/M/purush/male to "Male", female/F/stri/female to "Female". Do not infer from name.
- birthDate: convert any date format to YYYY-MM-DD; if only age given, set birthDate to null
- age: integer years only (e.g. "21 years old" → 21, "21" → 21)
- pinCode: exactly 6 digits, no spaces
- address: house/street/building address (NOT the area/locality name)
- area: locality/village/neighbourhood name (e.g. Kathwada, Navrangpura)
- symptoms: list of health complaints mentioned (e.g., ["fever", "cough"])
- doctor_name: extract doctor's name as mentioned (include "Dr." if present, e.g., "Dr. Sharma")
- appointment_slot: extract time/slot as mentioned (e.g., "10am", "2:30pm", "tomorrow 5pm")
- Return ONLY the JSON, no extra text

Current year: {year}

Text:
\"\"\"{text}\"\"\"
"""


# ── The Agent ─────────────────────────────────────────────────────────────────

class AppointmentAgent:

    # ── LLM call ─────────────────────────────────────────────────────────────

    async def _llm(self, messages: List[Dict], system: str) -> str:
        payload = {
            "model":   settings.LLM_MODEL,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream":  False,
            "options": {
                "temperature": settings.LLM_TEMPERATURE,
                "num_predict": settings.LLM_MAX_TOKENS,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(f"{settings.OLLAMA_BASE_URL}/api/chat", json=payload)
                r.raise_for_status()
                return r.json()["message"]["content"]
        except httpx.ConnectError:
            return "⚠️ Cannot connect to Ollama. Please ensure Ollama is running."
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "Sorry, something went wrong. Please try again."

    # ── LLM-based patient info extractor ─────────────────────────────────────

    async def _extract_patient_info(self, text: str) -> Dict:
        """
        Uses LLM to extract patient data from ANY natural language.
        Works for mixed Gujarati/Hindi/English.
        Example:
          "my name is panchal devangbhai hasmukhbhai and i leave in kathwada 382430
           and my contact is 8511274939 and i am 21 years old Male"
        →  {firstName: panchal, middleName: devangbhai, lastName: hasmukhbhai,
            mobile: 8511274939, gender: 1, age: 21, pinCode: 382430, area: kathwada}
        """
        # Use simple replace to avoid .format() interpreting braces inside the JSON snippet
        prompt = EXTRACTION_PROMPT.replace("{year}", str(datetime.now().year)).replace("{text}", text)
        payload = {
            "model":   settings.LLM_MODEL,
            "messages": [
                {"role": "system", "content": "Return ONLY valid JSON. No markdown. No explanation."},
                {"role": "user",   "content": prompt},
            ],
            "stream":  False,
            "options": {"temperature": 0.0, "num_predict": 512},
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(f"{settings.OLLAMA_BASE_URL}/api/chat", json=payload)
                r.raise_for_status()
                raw = r.json()["message"]["content"].strip()
                # Robustly find JSON even if LLM adds extra text
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                if m:
                    data = json.loads(m.group())
                    logger.info(f"[Extraction] Result: {data}")
                    # Process gender
                    if 'gender' in data and data['gender']:
                        if data['gender'].lower() == 'male':
                            data['gender'] = 1
                        elif data['gender'].lower() == 'female':
                            data['gender'] = 2
                        else:
                            data['gender'] = None
                    # Store doctor_name and appointment_slot for auto-matching
                    # Don't remove them - let the booking logic use them
                    return data
        except Exception as e:
            logger.error(f"[Extraction] Failed: {e}")
        return {}

    # ── Booking ───────────────────────────────────────────────────────────────

    async def _book_appointment(
        self, session_id: str, session: Dict, c: Dict
    ) -> Tuple[bool, Optional[Dict], str]:
        """
        Build complete AppointmentScheduleRequest from collected data and POST to API.
        Returns (success, booking_data, message_to_append).
        """
        try:
            # ── Birth date ────────────────────────────────────────────────────
            bd_raw = c.get("birthDate")
            if not bd_raw or bd_raw == 'null' or bd_raw == '':
                age = int(c.get("age", 25))
                approx_year = datetime.now().year - age
                bd_raw = f"{approx_year}-01-01"
            try:
                birth_dt = datetime.fromisoformat(bd_raw)
            except ValueError:
                try:
                    birth_dt = datetime.strptime(bd_raw, "%Y-%m-%d")
                except ValueError:
                    # If still invalid, use age fallback
                    age = int(c.get("age", 25))
                    approx_year = datetime.now().year - age
                    birth_dt = datetime(approx_year, 1, 1)

            # ── Patient ───────────────────────────────────────────────────────
            patient = Patient(
                firstName  = c.get("firstName", ""),
                middleName = c.get("middleName", ""),
                lastName   = c.get("lastName", ""),
                mobile     = c.get("mobile", ""),
                gender     = c.get("gender", 1),
                birthDate  = birth_dt,
                birthDateComponent = BirthDateComponent(
                    year=birth_dt.year, month=birth_dt.month, day=birth_dt.day
                ),
                healthId      = "",
                healthAddress = "",
                patientDetail = PatientDetail(
                    permanentAddress=PermanentAddress(
                        pinCode = c.get("pinCode", ""),
                        address = c.get("address", ""),
                        area    = c.get("area", ""),
                    )
                ),
            )

            # ── Appointment date/time with UTC conversion ──────────────────────
            sel_date = c.get("appointment_date", "")
            sel_time = c.get("appointment_time", "")
            appt_dt_utc_str = ""
            try:
                # Parse local datetime
                appt_dt_local = datetime.fromisoformat(f"{sel_date}T{sel_time}")
                # Convert to UTC string for API
                appt_dt_utc_str = appt_dt_local.isoformat() + "Z"
                logger.info(f"[Booking] DateTime: Local={sel_date}T{sel_time} → UTC={appt_dt_utc_str}")
                appt_dt = appt_dt_local
            except Exception as e:
                logger.error(f"DateTime parse error: {e}")
                appt_dt = datetime.now()
                appt_dt_utc_str = appt_dt.isoformat() + "Z"

            # ── AppointmentDetail ─────────────────────────────────────────────
            appointment_detail = AppointmentDetail(
                system               = 1,
                consultationType     = 1,
                slotDuration         = 0,
                externalId           = c.get("slot_external_id") or "",
                healthProfessionalId = c.get("health_professional_id") or "",
                facilityId           = c.get("facility_id") or settings.DEFAULT_FACILITY_ID or "",
                chiefComplaints      = c.get("symptoms") or [],
                appointentDateTime   = appt_dt,
            )

            booking_req = AppointmentScheduleRequest(
                patient=patient, appointmentDetail=appointment_detail
            )

            # ── Log full request body ─────────────────────────────────────────
            req_json = booking_req.model_dump_json(indent=2)
            logger.info(f"BOOKING REQUEST:\n{req_json}")
            print(f"\n{'='*80}\n🔵 BOOKING REQUEST BODY:\n{'='*80}\n{req_json}\n{'='*80}\n")

            result = await aarogya_api.schedule_appointment(booking_req)
            print(f"\n{'='*80}\n🟢 API RESPONSE:\n{'='*80}\n{result}\n{'='*80}\n")

            if result.get("success"):
                session["booked"] = True  # Mark as booked to prevent duplicates
                data    = result.get("data", {})
                booking_details = {**c, **data}  # Merge collected info with API response
                appt_id = data.get("appointmentId", "N/A")
                dr_name = c.get("doctor_name", "your doctor")
                msg = (
                    f"\n\n✅ **Appointment Confirmed!**\n"
                    f"🆔 Appointment ID : {appt_id}\n"
                    f"👨‍⚕️ Doctor         : {dr_name}\n"
                    f"📅 Date            : {sel_date}\n"
                    f"⏰ Time            : {c.get('slot_display', sel_time)}\n"
                    f"👤 Patient         : {c.get('firstName','')} {c.get('lastName','')}\n"
                    f"📞 Mobile          : {c.get('mobile','')}\\n"
                    f"🕐 UTC DateTime    : {appt_dt_utc_str}"
                )
                logger.info(f"[Booking Success] Appointment ID: {appt_id} | UTC: {appt_dt_utc_str}")
                return True, booking_details, msg
            else:
                err = result.get("error", "Unknown error")
                return False, None, f"\n\n❌ Booking failed: {err}"

        except Exception as e:
            logger.error(f"Booking exception: {e}", exc_info=True)
            return False, None, f"\n\n❌ Booking failed: {e}"

    # ── Main chat handler ─────────────────────────────────────────────────────

    async def chat(self, session_id: str, user_message: str) -> Dict[str, Any]:
        session   = session_manager.get(session_id)
        collected = session_manager.collected(session_id)

        logger.info(f"[{session_id}] stage={session['stage']} | user: {user_message[:100]}")

        msg_lower = user_message.lower()

        # ── STEP 1: Detect doctor + slot selection ────────────────────────────
        selection = detect_doctor_slot_selection(user_message, session["doctors"])
        if selection and session["stage"] in ("doctors_shown", "start"):
            doc  = selection["doctor"]
            slot = selection["slot"]
            logger.info(f"Doctor selected: {doc.get('healthProfessionalName')} | Slot: {slot.get('from')}-{slot.get('to')}")

            # Determine externalId from slot - prioritize and create unique ID
            ext_id = slot.get("externalId") or slot.get("slotId") or slot.get("id")
            
            # If still no ID, create one from slot data
            if not ext_id:
                slot_from = slot.get("from", slot.get("startTime", ""))
                slot_to = slot.get("to", slot.get("endTime", ""))
                ext_id = f"slot_{slot_from.replace(':', '')}_{slot_to.replace(':', '')}"
                logger.info(f"Generated slotId: {ext_id}")
            
            session_manager.update_collected(session_id, {
                "health_professional_id": doc.get("healthProfessionalId", ""),
                "doctor_name":            doc.get("healthProfessionalName", ""),
                "facility_id":            doc.get("facilityId") or settings.DEFAULT_FACILITY_ID or "",
                "appointment_date":       doc.get("appointmentDate", ""),
                "appointment_time":       slot.get("from", ""),
                "slot_external_id":       ext_id,
                "slot_display":           f"{slot.get('from','')} - {slot.get('to','')}",
            })
            session["stage"] = "selected"

        # Fallback: user may say "confirm Dr Dhruv Barot" or "conform Dr Dhruv Barot afternoon"
        # If selection not detected, try to detect explicit confirmation by doctor name
        if not selection and session["stage"] in ("doctors_shown", "start"):
            if re.search(r'\b(confirm|conform|book|yes|confirming)\b', user_message.lower()):
                # try to find a doctor by name mentioned in message
                found = None
                for d in session.get("doctors", []):
                    name = (d.get("healthProfessionalName") or d.get("name") or "").lower()
                    if name and name in user_message.lower():
                        found = d
                        break

                if found:
                    # determine preferred time if mentioned (morning/afternoon/evening) or explicit time
                    prefer = None
                    if "afternoon" in user_message.lower():
                        prefer = "afternoon"
                    elif "morning" in user_message.lower():
                        prefer = "morning"
                    elif "evening" in user_message.lower():
                        prefer = "evening"

                    # try to extract explicit time like 10:00, 10:00am, 10am
                    time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', user_message.lower())
                    time_str = None
                    if time_match:
                        hh = int(time_match.group(1))
                        mm = int(time_match.group(2) or 0)
                        ampm = time_match.group(3)
                        if ampm == 'pm' and hh != 12:
                            hh += 12
                        if ampm == 'am' and hh == 12:
                            hh = 0
                        time_str = f"{hh:02d}:{mm:02d}"

                    # pick a slot matching preference
                    all_slots = _flatten_available_slots(found, filter_today_future=True)
                    chosen_slot = None
                    if time_str:
                        for s in all_slots:
                            s_from = (s.get('from') or s.get('startTime') or '').strip().lower()
                            if time_str in s_from:
                                chosen_slot = s
                                break
                    if not chosen_slot and prefer:
                        for s in all_slots:
                            s_from = (s.get('from') or s.get('startTime') or '').strip().lower()
                            try:
                                h = int(s_from.split(':')[0])
                            except Exception:
                                h = None
                            if prefer == 'morning' and h is not None and 6 <= h < 12:
                                chosen_slot = s
                                break
                            if prefer == 'afternoon' and h is not None and 12 <= h < 17:
                                chosen_slot = s
                                break
                            if prefer == 'evening' and h is not None and 17 <= h < 22:
                                chosen_slot = s
                                break

                    # fallback to first available
                    if not chosen_slot and all_slots:
                        chosen_slot = all_slots[0]

                    if chosen_slot:
                        ext_id = chosen_slot.get('slotId') or chosen_slot.get('externalId') or chosen_slot.get('id')
                        
                        # If still no ID, create one from slot data
                        if not ext_id:
                            slot_from = chosen_slot.get('from', chosen_slot.get('startTime', ''))
                            slot_to = chosen_slot.get('to', chosen_slot.get('endTime', ''))
                            ext_id = f"slot_{slot_from.replace(':', '')}_{slot_to.replace(':', '')}"
                            logger.info(f"Generated slotId for confirm: {ext_id}")
                        
                        session_manager.update_collected(session_id, {
                            "health_professional_id": found.get("healthProfessionalId", ""),
                            "doctor_name": found.get("healthProfessionalName", ""),
                            "facility_id": found.get("facilityId") or settings.DEFAULT_FACILITY_ID or "",
                            "appointment_date": found.get("appointmentDate", ""),
                            "appointment_time": chosen_slot.get("from", chosen_slot.get('startTime', '')),
                            "slot_external_id": ext_id,
                            "slot_display": f"{chosen_slot.get('from','')} - {chosen_slot.get('to','')}",
                        })
                        session["stage"] = "selected"
            # Safe logging: use collected values to avoid referencing local vars that may not exist
            collected_now = session_manager.collected(session_id)
            doctor_id = collected_now.get("health_professional_id") or collected_now.get("selectedDoctorId")
            slot_ext = collected_now.get("slot_external_id") or collected_now.get("selectedSlotExternalId")
            logger.info(f"Stage → selected | doctor={doctor_id} slot_ext={slot_ext}")

        # If still not selected, but we have a health_professional_id (static/default), try to fetch doctor by id
        if session.get("stage") != "selected":
            collected_now = session_manager.collected(session_id)
            hp_id = collected_now.get("health_professional_id") or collected_now.get("selectedDoctorId")
            if hp_id:
                # ensure we have doctors cached in session
                docs = session.get("doctors") or await doctors_cache.get_doctors()
                # find doctor by id
                target = None
                for d in docs:
                    if d.get("healthProfessionalId") == hp_id:
                        target = d
                        break

                if target:
                    all_slots = _flatten_available_slots(target, filter_today_future=True)
                    # try to parse a time from user's message
                    time_match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", user_message.lower())
                    time_str = None
                    if time_match:
                        hh = int(time_match.group(1))
                        mm = int(time_match.group(2) or 0)
                        ampm = time_match.group(3)
                        if ampm == 'pm' and hh != 12:
                            hh += 12
                        if ampm == 'am' and hh == 12:
                            hh = 0
                        time_str = f"{hh:02d}:{mm:02d}"

                    chosen_slot = None
                    if time_str:
                        for s in all_slots:
                            s_from = (s.get('from') or s.get('startTime') or '').strip().lower()
                            if time_str in s_from:
                                chosen_slot = s
                                break
                    if not chosen_slot and all_slots:
                        chosen_slot = all_slots[0]

                    if chosen_slot:
                        ext_id = chosen_slot.get('slotId') or chosen_slot.get('externalId') or chosen_slot.get('id')
                        
                        # If still no ID, create one from slot data
                        if not ext_id:
                            slot_from = chosen_slot.get('from', chosen_slot.get('startTime', ''))
                            slot_to = chosen_slot.get('to', chosen_slot.get('endTime', ''))
                            ext_id = f"slot_{slot_from.replace(':', '')}_{slot_to.replace(':', '')}"
                            logger.info(f"Generated slotId for hp_id: {ext_id}")
                        
                        session_manager.update_collected(session_id, {
                            "health_professional_id": target.get("healthProfessionalId", ""),
                            "doctor_name": target.get("healthProfessionalName", ""),
                            "facility_id": target.get("facilityId") or settings.DEFAULT_FACILITY_ID or "",
                            "appointment_date": target.get("appointmentDate", ""),
                            "appointment_time": chosen_slot.get("from", chosen_slot.get('startTime', '')),
                            "slot_external_id": ext_id,
                            "slot_display": f"{chosen_slot.get('from','')} - {chosen_slot.get('to','')}",
                        })
                        session["stage"] = "selected"
                        logger.info(f"Auto-selected doctor by id {hp_id} and chosen slot {ext_id}")

        # ── STEP 2: LLM extraction of patient info ────────────────────────────
        # Run whenever message might contain personal details
        PERSONAL_HINTS = [
            "name", "naam", "mobile", "contact", "number", "live", "rehta", "raheta",
            "address", "pincode", "pin", "area", "gender", "male", "female", "purush",
            "stri", "old", "born", "years", "age", "my ", "maru", "mara", "hun ",
            "meri", "mera", "i am", "i live", "hu ", "ane", "and my", "aapo",
            "dr", "doctor", "time", "slot", "am", "pm", "morning", "afternoon", "evening",
            "book", "appointment", "schedule",
        ]
        might_have_info = any(hint in msg_lower for hint in PERSONAL_HINTS)

        if might_have_info or session["stage"] == "selected":
            extracted = await self._extract_patient_info(user_message)
            clean = {k: v for k, v in extracted.items() if v is not None and str(v).strip() != ""}
            if clean:
                # ── AUTO-SELECT DOCTOR BY NAME AND APPOINTMENT SLOT ──────────
                extracted_doctor_name = clean.get("doctor_name")
                extracted_slot_time = clean.get("appointment_slot")
                
                logger.info(f"[Auto-select] doctor_name='{extracted_doctor_name}' slot='{extracted_slot_time}' stage='{session['stage']}'")
                
                if extracted_doctor_name and extracted_slot_time and session["stage"] in ("start", "doctors_shown"):
                    # Ensure we have doctors cached
                    docs = session.get("doctors") or await doctors_cache.get_doctors()
                    session["doctors"] = docs
                    
                    logger.info(f"[Auto-select] Loaded {len(docs)} doctors")
                    
                    # Find doctor by name
                    found_doctor = find_doctor_by_name(extracted_doctor_name, docs)
                    
                    logger.info(f"[Auto-select] Found doctor: {found_doctor.get('healthProfessionalName') if found_doctor else None}")
                    
                    if found_doctor:
                        # Select slot by time
                        selected_slot = select_slot_by_time(found_doctor, extracted_slot_time)
                        
                        logger.info(f"[Auto-select] Selected slot: {selected_slot.get('from') if selected_slot else None}")
                        
                        if selected_slot:
                            ext_id = selected_slot.get('slotId') or selected_slot.get('externalId') or selected_slot.get('id')
                            
                            # If still no ID, create one from slot data
                            if not ext_id:
                                slot_from = selected_slot.get('from', selected_slot.get('startTime', ''))
                                slot_to = selected_slot.get('to', selected_slot.get('endTime', ''))
                                ext_id = f"slot_{slot_from.replace(':', '')}_{slot_to.replace(':', '')}"
                                logger.info(f"Generated slotId for auto-select: {ext_id}")
                            
                            session_manager.update_collected(session_id, {
                                "health_professional_id": found_doctor.get("healthProfessionalId", ""),
                                "facility_id": found_doctor.get("facilityId") or settings.DEFAULT_FACILITY_ID or "",
                                "appointment_date": found_doctor.get("appointmentDate", ""),
                                "appointment_time": selected_slot.get("from", selected_slot.get('startTime', '')),
                                "slot_external_id": ext_id,
                                "slot_display": f"{selected_slot.get('from','')} - {selected_slot.get('to','')}",
                            })
                            session["stage"] = "selected"
                            logger.info(f"✅ Auto-selected doctor '{extracted_doctor_name}' and slot '{extracted_slot_time}'")
                            doctor_selected_msg = f"✅ Doctor '{found_doctor.get('healthProfessionalName')}' selected with slot {selected_slot.get('from', '')} - {selected_slot.get('to', '')}"
                            session_manager.add_message(session_id, "system", doctor_selected_msg)
                
                # Remove doctor/slot fields from clean before storing patient data
                clean.pop("doctor_name", None)
                clean.pop("appointment_slot", None)
                clean.pop("symptoms", None)  # Also remove symptoms as it's handled separately
                
                if clean:
                    session_manager.update_collected(session_id, clean)
                    logger.info(f"[Extraction] Stored patient data: {clean}")

        # ── STEP 3: Symptom detection → fetch doctors ─────────────────────────
        SYMPTOM_KWS = [
            "fever", "pain", "chest pain", "headache", "cough", "cold",
            "heart", "nausea", "vomiting", "migraine", "skin", "rash",
            "tav", "dard", "bukhar",
        ]
        symptoms_found = [kw for kw in SYMPTOM_KWS if kw in msg_lower]
        
        # Also include symptoms extracted by LLM
        collected_now = session_manager.collected(session_id)
        extracted_symptoms = collected_now.get("symptoms", [])
        if extracted_symptoms:
            if isinstance(extracted_symptoms, list):
                symptoms_found.extend(extracted_symptoms)
            else:
                symptoms_found.append(str(extracted_symptoms))
        
        # Remove duplicates
        symptoms_found = list(set(symptoms_found))

        if symptoms_found and not session["doctors"] and session["stage"] == "start":
            logger.info(f"Symptoms detected: {symptoms_found}")
            all_docs = await doctors_cache.get_doctors()
            filtered  = filter_doctors_by_symptoms(all_docs, symptoms_found)
            session["doctors"] = filtered
            session_manager.update_collected(session_id, {"symptoms": symptoms_found})
            doc_text = format_doctors_for_display(filtered)
            session_manager.add_message(session_id, "system", f"[Available doctors fetched from API]\n{doc_text}")
            session["stage"] = "doctors_shown"

        # ── STEP 4: Add user message to history ───────────────────────────────
        session_manager.add_message(session_id, "user", user_message)

        # ── STEP 5: Check if ready to book ───────────────────────────────────
        collected        = session_manager.collected(session_id)
        doctor_selected  = bool(collected.get("health_professional_id"))
        missing_fields   = missing_patient_fields(collected)
        booking_complete = False
        booking_details  = None
        booking_msg      = ""

        if session.get("booked", False):
            logger.info("❌ Appointment already booked for this session. Skipping duplicate booking.")
            booking_complete = True
            booking_details = collected
            booking_msg = "\n\n⚠️ Appointment already booked in this session. Cannot book again."
        elif doctor_selected and not missing_fields:
            logger.info("✅ All fields present — triggering booking!")
            booking_complete, booking_details, booking_msg = await self._book_appointment(
                session_id, session, collected
            )
        else:
            logger.info(f"Doctor selected={doctor_selected} | Missing={missing_fields}")

        # ── STEP 6: Generate LLM response ─────────────────────────────────────
        context_str = build_context_summary(collected)
        missing_str = ", ".join(missing_fields) if missing_fields else "None — ready to book!"
        system = SYSTEM_PROMPT.format(context=context_str, missing=missing_str)

        llm_out = await self._llm(session_manager.messages(session_id), system)

        # Append booking result to LLM response
        if booking_msg:
            llm_out += booking_msg

        session_manager.add_message(session_id, "assistant", llm_out)
        logger.info(f"[{session_id}] Chat done. Booked={booking_complete}")

        return {
            "session_id":        session_id,
            "response":          llm_out,
            "appointment_booked": booking_complete,
            "booking_details":   booking_details or collected,
        }

    def reset(self, session_id: str):
        session_manager.reset(session_id)


appointment_agent = AppointmentAgent()