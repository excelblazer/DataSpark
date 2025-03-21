from functools import wraps
import redis
from app.config import settings

redis_client = redis.from_url(settings.REDIS_URL) if settings.REDIS_URL else None

def cache_result(expire_time: int = 300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not redis_client:
                return await func(*args, **kwargs)
                
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            cached_result = redis_client.get(cache_key)
            
            if cached_result:
                return cached_result
                
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expire_time, result)
            return result
        return wrapper
    return decorator
