from fastapi import APIRouter, Depends
from ..dependencies_util import get_current_user
from ..models import User

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=dict)
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "name": current_user.name,
        "surname": current_user.surname,
        "email": current_user.email,
        "role": current_user.role,
        "created_at": getattr(current_user, "created_at", None)
    }
