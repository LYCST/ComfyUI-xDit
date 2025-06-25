"""
Ray Manager
==========

Ray cluster management and initialization for xDiT multi-GPU acceleration.
"""

import os
import logging
import time
from typing import Dict, Any, Optional, List
import subprocess

# Try to import Ray
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class RayManager:
    """
    Manages Ray cluster for xDiT multi-GPU acceleration.
    """
    
    def __init__(self):
        self.is_initialized = False
        self.cluster_config = {}
        self.available_gpus = []
        
    def initialize(self, 
                  num_gpus: int = None,
                  memory_limit_gb: int = None,
                  object_store_memory_gb: int = None,
                  temp_dir: str = None,
                  dashboard_port: int = 8265,
                  dashboard_host: str = "0.0.0.0") -> bool:
        """
        Initialize Ray cluster
        
        Args:
            num_gpus: Number of GPUs to use (None for auto-detect)
            memory_limit_gb: Memory limit in GB (None for auto-detect)
            object_store_memory_gb: Object store memory in GB (None for auto-detect)
            temp_dir: Temporary directory for Ray
            dashboard_port: Ray dashboard port
            dashboard_host: Ray dashboard host
        """
        try:
            if not RAY_AVAILABLE:
                logger.error("Ray is not available")
                return False
            
            if self.is_initialized:
                logger.info("Ray already initialized")
                return True
            
            # Auto-detect GPUs if not specified
            if num_gpus is None:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    num_gpus = torch.cuda.device_count()
                else:
                    num_gpus = 0
            
            # Auto-detect memory if not specified
            if memory_limit_gb is None:
                import psutil
                total_memory = psutil.virtual_memory().total / (1024**3)  # GB
                memory_limit_gb = int(total_memory * 0.8)  # Use 80% of system memory
            
            if object_store_memory_gb is None:
                object_store_memory_gb = min(4, memory_limit_gb // 4)  # 4GB or 25% of memory
            
            # Set up temp directory
            if temp_dir is None:
                temp_dir = os.path.join(os.getcwd(), "ray_temp")
            
            os.makedirs(temp_dir, exist_ok=True)
            
            # Configure Ray
            ray_config = {
                "num_cpus": os.cpu_count(),
                "num_gpus": num_gpus,
                "object_store_memory": object_store_memory_gb * 1024 * 1024 * 1024,  # Convert to bytes
                "dashboard_port": dashboard_port,
                "dashboard_host": dashboard_host,
                "ignore_reinit_error": True,
                "local_mode": False,
                "log_to_driver": True,
                "logging_level": logging.INFO,
            }
            
            # Initialize Ray
            logger.info(f"Initializing Ray with config: {ray_config}")
            ray.init(**ray_config)
            
            # Verify initialization
            if ray.is_initialized():
                self.is_initialized = True
                self.cluster_config = ray_config
                
                # Get available resources
                resources = ray.available_resources()
                logger.info(f"Ray initialized successfully")
                logger.info(f"Available resources: {resources}")
                
                # Get GPU information
                self._update_gpu_info()
                
                return True
            else:
                logger.error("Failed to initialize Ray")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing Ray: {e}")
            return False
    
    def _update_gpu_info(self):
        """Update GPU information"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.available_gpus = list(range(torch.cuda.device_count()))
                logger.info(f"Available GPUs: {self.available_gpus}")
            else:
                self.available_gpus = []
                logger.warning("No CUDA GPUs available")
        except Exception as e:
            logger.error(f"Error updating GPU info: {e}")
            self.available_gpus = []
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster information"""
        if not self.is_initialized:
            return {"error": "Ray not initialized"}
        
        try:
            resources = ray.available_resources()
            nodes = ray.nodes()
            
            return {
                "is_initialized": self.is_initialized,
                "available_resources": resources,
                "nodes": len(nodes),
                "available_gpus": self.available_gpus,
                "dashboard_url": f"http://localhost:{self.cluster_config.get('dashboard_port', 8265)}"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def shutdown(self):
        """Shutdown Ray cluster"""
        try:
            if self.is_initialized and ray.is_initialized():
                ray.shutdown()
                self.is_initialized = False
                logger.info("Ray cluster shutdown")
        except Exception as e:
            logger.error(f"Error shutting down Ray: {e}")
    
    def restart(self) -> bool:
        """Restart Ray cluster"""
        try:
            self.shutdown()
            time.sleep(2)  # Wait for shutdown
            return self.initialize()
        except Exception as e:
            logger.error(f"Error restarting Ray: {e}")
            return False
    
    def get_worker_placement_group(self, num_gpus: int = 1) -> Optional[Any]:
        """Create a placement group for workers"""
        try:
            if not self.is_initialized:
                logger.error("Ray not initialized")
                return None
            
            # Create placement group for GPU workers
            pg = ray.util.placement_group(
                bundles=[{"GPU": 1} for _ in range(num_gpus)],
                strategy="STRICT_PACK"
            )
            
            # Wait for placement group to be ready
            ray.get(pg.ready())
            logger.info(f"Created placement group with {num_gpus} GPU bundles")
            
            return pg
            
        except Exception as e:
            logger.error(f"Error creating placement group: {e}")
            return None

# Global Ray manager instance
ray_manager = RayManager()

def initialize_ray(**kwargs) -> bool:
    """Initialize Ray cluster"""
    return ray_manager.initialize(**kwargs)

def get_ray_info() -> Dict[str, Any]:
    """Get Ray cluster information"""
    return ray_manager.get_cluster_info()

def shutdown_ray():
    """Shutdown Ray cluster"""
    ray_manager.shutdown()

def is_ray_available() -> bool:
    """Check if Ray is available and initialized"""
    return RAY_AVAILABLE and ray_manager.is_initialized 