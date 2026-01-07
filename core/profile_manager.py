"""Profile Manager"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_PROFILE = {
    "name": "User",
    "role": "Owner",
    "preferences": {
        "brevity": "normal",
        "tone": "helpful",
        "tech_stack": []
    },
    "bio": "I am the owner of this AI assistant."
}

class UserProfile(BaseModel):
    """User profile schema."""
    name: str = "User"
    role: str = "Owner"
    preferences: Dict[str, Any] = {}
    bio: str = ""
    
    class Config:
        extra = "allow"

class ProfileManager:
    """Manages the single-user profile."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.profile_path = self.data_dir / "user_profile.json"
        self._ensure_data_dir()
    
    def _ensure_data_dir(self):
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
    def get_profile(self) -> Dict[str, Any]:
        """Get the current user profile."""
        if not self.profile_path.exists():
            return DEFAULT_PROFILE.copy()
            
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load profile: {e}")
            return DEFAULT_PROFILE.copy()
    
    def update_profile(self, data: Dict[str, Any]) -> UserProfile:
        """Update the user profile."""
        current = self.get_profile()
        
        updated = {**current, **data}
        
        if "preferences" in data and "preferences" in current:
            updated["preferences"] = {**current["preferences"], **data["preferences"]}
        
        try:
            profile = UserProfile(**updated)
            
            with open(self.profile_path, "w", encoding="utf-8") as f:
                json.dump(profile.model_dump(), f, indent=2)
            
            logger.info("User profile updated")
            return profile
            
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
            return UserProfile(**updated)

# Global instance
_profile_manager = None

def get_profile_manager() -> ProfileManager:
    """Get the global ProfileManager instance."""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = ProfileManager()
    return _profile_manager
