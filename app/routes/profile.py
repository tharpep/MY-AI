"""Profile API routes - User profile management"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Body

from core.profile_manager import get_profile_manager, UserProfile

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/profile")
async def get_profile() -> Dict[str, Any]:
    """
    Get the current user profile.
    
    Returns:
        User profile object
    """
    try:
        manager = get_profile_manager()
        return manager.get_profile()
    except Exception as e:
        logger.error(f"Failed to get profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve profile")


@router.patch("/profile")
async def update_profile(
    profile_data: Dict[str, Any] = Body(..., description="Partial profile update")
) -> UserProfile:
    """
    Update the user profile.
    
    Args:
        profile_data: Dictionary of fields to update
        
    Returns:
        Updated profile object
    """
    try:
        manager = get_profile_manager()
        updated = manager.update_profile(profile_data)
        return updated
    except Exception as e:
        logger.error(f"Failed to update profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to update profile")
