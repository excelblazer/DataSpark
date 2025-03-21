from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import os
from .config import settings

# Database configuration
SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL

# Determine if we need SSL configuration
ssl_config = {}

if settings.DATABASE_USE_SSL:
    ssl_config = {
        "ssl": {
            "ca": settings.DATABASE_SSL_CA,
            "cert": settings.DATABASE_SSL_CERT,
            "key": settings.DATABASE_SSL_KEY
        }
    }

connect_args = {"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {}
if settings.DATABASE_USE_SSL:
    connect_args.update(ssl_config)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args=connect_args,
    poolclass=StaticPool
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
