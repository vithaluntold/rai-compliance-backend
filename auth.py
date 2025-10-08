from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv

from database import SessionLocal
from models import User

load_dotenv()

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer(auto_error=False)


class AuthManager:
    def __init__(self):
        self.pwd_context = pwd_context
        self.secret_key = SECRET_KEY
        self.algorithm = ALGORITHM
        self.access_token_expire_minutes = ACCESS_TOKEN_EXPIRE_MINUTES

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return self.pwd_context.hash(password)

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create a JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[dict]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            return None

    def authenticate_user(self, db: Session, email: str, password: str) -> Optional[User]:
        """Authenticate a user with email and password"""
        user = db.query(User).filter(User.email == email).first()
        
        if not user:
            return None
        
        if not self.verify_password(password, user.hashed_password):
            return None
        
        return user

    def get_current_user(
        self, 
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
        db: Session = Depends(lambda: next(SessionLocal()))
    ) -> Optional[User]:
        """Get current authenticated user from JWT token"""
        if not credentials:
            return None
        
        token_data = self.verify_token(credentials.credentials)
        
        if not token_data:
            return None
        
        user_email = token_data.get("sub")
        if not user_email:
            return None
        
        user = db.query(User).filter(User.email == user_email).first()
        return user

    def require_auth(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: Session = Depends(lambda: next(SessionLocal()))
    ) -> User:
        """Require authentication - raises HTTPException if not authenticated"""
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        token_data = self.verify_token(credentials.credentials)
        
        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_email = token_data.get("sub")
        if not user_email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user = db.query(User).filter(User.email == user_email).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )
        
        return user

    def require_role(self, required_role: str):
        """Decorator to require specific role"""
        def role_checker(current_user: User = Depends(self.require_auth)):
            if current_user.role != required_role and current_user.role != "admin":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            return current_user
        return role_checker

    def require_admin(self):
        """Require admin role"""
        def admin_checker(current_user: User = Depends(self.require_auth)):
            if current_user.role != "admin":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin access required"
                )
            return current_user
        return admin_checker


# Create singleton instance
auth_manager = AuthManager()

# Convenience dependencies
get_current_user = auth_manager.get_current_user
require_auth = auth_manager.require_auth
require_admin = auth_manager.require_admin()