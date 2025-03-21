import pytest
from fastapi.testclient import TestClient
import os
import sys
import jwt
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app
from app.config import settings

client = TestClient(app)

def create_test_token(data: dict, expires_delta: timedelta = None):
    """Create a test JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

@pytest.fixture
def admin_token():
    """Create an admin token for testing"""
    return create_test_token({"sub": "admin", "role": "admin"})

@pytest.fixture
def user_token():
    """Create a regular user token for testing"""
    return create_test_token({"sub": "user", "role": "viewer"})

def test_read_main():
    """Test that the main endpoint returns 200"""
    response = client.get("/")
    assert response.status_code == 200

def test_login_success():
    """Test successful login"""
    response = client.post(
        "/token",
        data={"username": "admin", "password": "password"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_login_failure():
    """Test failed login"""
    response = client.post(
        "/token",
        data={"username": "admin", "password": "wrong-password"}
    )
    assert response.status_code == 401

def test_protected_endpoint_with_token(admin_token):
    """Test accessing a protected endpoint with a valid token"""
    response = client.get(
        "/api/v1/warehouse/datasets",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200

def test_protected_endpoint_without_token():
    """Test accessing a protected endpoint without a token"""
    response = client.get("/api/v1/warehouse/datasets")
    assert response.status_code == 401

def test_protected_endpoint_with_invalid_token():
    """Test accessing a protected endpoint with an invalid token"""
    response = client.get(
        "/api/v1/warehouse/datasets",
        headers={"Authorization": "Bearer invalid-token"}
    )
    assert response.status_code == 401

def test_user_access_admin_endpoint(user_token):
    """Test a regular user trying to access an admin endpoint"""
    response = client.post(
        "/api/v1/data/upload",
        headers={"Authorization": f"Bearer {user_token}"},
        files={"file": ("test.csv", "a,b,c\n1,2,3", "text/csv")}
    )
    assert response.status_code == 403  # Forbidden
