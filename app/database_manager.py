from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from tenacity import retry, stop_after_attempt, wait_exponential

class DatabaseManager:
    def __init__(self, database_url: str, pool_size: int = 5):
        self.database_url = database_url
        self.engine = self._create_engine(pool_size)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def _create_engine(self, pool_size: int):
        return create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def get_connection(self):
        return self.engine.connect()
    
    def get_session(self):
        return self.SessionLocal()
