import os
import redis
from dotenv import load_dotenv

load_dotenv()

_pool = None


def get_redis():
    """Return a Redis client backed by a shared connection pool."""
    global _pool
    if _pool is None:
        _pool = redis.ConnectionPool(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
        )
    return redis.Redis(connection_pool=_pool)
