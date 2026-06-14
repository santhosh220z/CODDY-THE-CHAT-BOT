from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from backend.database.connection import get_db
from backend.models.models import User
from backend.auth.security import get_password_hash, verify_password, create_access_token, decode_access_token
from backend.utils.logging import logger

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

# Pydantic Schemas
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    role: Optional[str] = "user" # user, admin

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    email: str

class UserProfile(BaseModel):
    id: int
    email: str
    role: str

# Dependency to get current user
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception
    
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    
    email: str = payload.get("sub")
    if email is None:
        raise credentials_exception
        
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

# Dependency to check if user is admin
def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Hash password and save
    hashed_pwd = get_password_hash(user_data.password)
    # The first user registered or specific domains could be forced to admin, or allow manual specification for local testing
    # Let's seed the first user as admin if there are no users in DB.
    user_count = db.query(User).count()
    role = "admin" if user_count == 0 else user_data.role
    if role not in ["user", "admin"]:
        role = "user"

    new_user = User(
        email=user_data.email,
        hashed_password=hashed_pwd,
        role=role
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    logger.info(f"Registered user: {new_user.email} with role: {new_user.role}")
    return {"message": "Registration successful", "email": new_user.email, "role": new_user.role}

@router.post("/login", response_model=Token)
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create JWT token
    access_token = create_access_token(data={"sub": user.email, "role": user.role})
    logger.info(f"User logged in successfully: {user.email}")
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user.role,
        "email": user.email
    }

@router.post("/logout")
def logout():
    # Stateless JWT logout is handled on client side, but we provide endpoint for consistency
    return {"message": "Logged out successfully"}

@router.get("/profile", response_model=UserProfile)
def profile(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "role": current_user.role
    }
