"""
Redis Queue Infrastructure
Generic queue management for async job processing
"""

import logging
from typing import Optional
from dataclasses import dataclass

from arq import create_pool
from arq.connections import RedisSettings, ArqRedis
from arq.jobs import Job

logger = logging.getLogger(__name__)


@dataclass
class JobStatus:
    """Status of a queued job"""
    job_id: str
    status: str  # queued, processing, completed, failed
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


class RedisQueue:
    """
    Generic Redis queue manager using arq.
    
    Provides enqueueing and job status tracking.
    """
    
    def __init__(self, redis_settings: Optional[RedisSettings] = None):
        """
        Initialize the queue manager.
        
        Args:
            redis_settings: Redis connection settings. Defaults to localhost:6379
        """
        self.settings = redis_settings or RedisSettings(host='localhost', port=6379)
        self._pool: Optional[ArqRedis] = None
    
    async def get_pool(self) -> ArqRedis:
        """Get or create Redis connection pool"""
        if self._pool is None:
            self._pool = await create_pool(self.settings)
        return self._pool
    
    async def enqueue(self, function_name: str, *args, **kwargs) -> str:
        """
        Enqueue a job for processing.
        
        Args:
            function_name: Name of the worker function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            job_id: The job identifier for tracking
        """
        pool = await self.get_pool()
        job = await pool.enqueue_job(function_name, *args, **kwargs)
        logger.info(f"Enqueued job {job.job_id} for {function_name}")
        return job.job_id
    
    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """
        Get the status of a job.
        
        Args:
            job_id: The job identifier
            
        Returns:
            JobStatus or None if not found
        """
        try:
            pool = await self.get_pool()
            job = Job(job_id, pool)
            
            status = await job.status()
            info = await job.info()
            
            if info is None:
                return None
            
            # Map arq JobStatus enum to our status strings
            status_name = status.name if hasattr(status, 'name') else str(status)
            status_map = {
                'deferred': 'queued',
                'queued': 'queued', 
                'in_progress': 'processing',
                'complete': 'completed',
                'not_found': 'not_found',
            }
            
            # Safely get attributes from JobDef (different versions may have different attrs)
            enqueue_time = getattr(info, 'enqueue_time', None)
            # JobDef doesn't have finish_time, we check if result exists for completion
            
            # Check if job failed (result is exception)
            error_msg = None
            result = getattr(info, 'result', None)
            if result is not None and isinstance(result, Exception):
                error_msg = str(result)
            
            return JobStatus(
                job_id=job_id,
                status=status_map.get(status_name, 'unknown'),
                created_at=enqueue_time.isoformat() if enqueue_time else '',
                completed_at=None,  # JobDef doesn't track this
                error=error_msg
            )
        except Exception as e:
            logger.error(f"Error getting job status for {job_id}: {e}")
            raise
    
    async def close(self):
        """Close the Redis connection pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None


# Singleton instance
_queue: Optional[RedisQueue] = None


async def get_redis_queue() -> RedisQueue:
    """Get the global RedisQueue instance"""
    global _queue
    if _queue is None:
        _queue = RedisQueue()
    return _queue
