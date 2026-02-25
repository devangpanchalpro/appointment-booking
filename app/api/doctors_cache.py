"""
Doctor availability cache
Fetches from external API and stores in memory
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from app.api.external_client import aarogya_api
from app.config.settings import settings

logger = logging.getLogger(__name__)


class DoctorsCache:
    """In-memory cache for doctor availability"""
    
    def __init__(self):
        self._doctors: List[Dict] = []
        self._last_fetched: datetime = None
        self._cache_duration = timedelta(minutes=15)  # Refresh every 15 min
    
    async def get_doctors(
        self,
        facility_id: Optional[str] = None,
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get doctors with availability.
        Uses cache if recent, otherwise fetches from API.
        """
        fac_id = facility_id or settings.DEFAULT_FACILITY_ID
        
        # Check if cache valid
        if not force_refresh and self._doctors and self._last_fetched:
            age = datetime.now() - self._last_fetched
            if age < self._cache_duration:
                logger.info(f"Using cached doctors (age: {age.seconds}s)")
                return self._doctors
        
        # Fetch from API
        logger.info("Fetching doctors from Aarogya API...")
        doctors = await aarogya_api.get_doctors_availability(facility_id=fac_id)
        
        if doctors:
            self._doctors = doctors
            self._last_fetched = datetime.now()
            logger.info(f"Cached {len(doctors)} doctors")
        
        return doctors
    
    async def search_by_specialization(self, specialization: str) -> List[Dict]:
        """Filter cached doctors by specialization"""
        all_doctors = await self.get_doctors()
        spec_lower = specialization.lower()
        
        return [
            doc for doc in all_doctors
            if spec_lower in doc.get("specialization", "").lower()
        ]


# Singleton
doctors_cache = DoctorsCache()