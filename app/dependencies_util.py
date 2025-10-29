from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from .models import User
from .dependencies.auth import get_current_user

security = HTTPBearer()


def require_role(required_role: str):
    """
    Dependency factory that ensures current user has the given role.
    Example: require_role("teacher")
    """

    def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: {required_role} role required"
            )
        return current_user

    return role_checker
