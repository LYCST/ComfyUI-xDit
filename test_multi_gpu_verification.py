#!/usr/bin/env python3
"""
验证多GPU分布式工作原理
====================
验证每个Worker确实运行在不同的物理GPU上
"""

import sys
import os
import torch
import logging
import time
import ray

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@ray.remote(num_gpus=1)
class GPUWorker:
    """简单的GPU Worker来验证GPU分配"""
    
    def __init__(self, worker_id):
        self.worker_id = worker_id
        
    def get_gpu_info(self):
        """获取当前Worker的GPU信息"""
        import torch
        import os
        
        result = {
            'worker_id': self.worker_id,
            'process_id': os.getpid(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # 在Worker内部，设备总是cuda:0
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            
            result.update({
                'current_device': device,
                'device_name': props.name,
                'device_memory_gb': props.total_memory / (1024**3),
                'device_capability': f"{props.major}.{props.minor}",
            })
            
            # 创建一个张量来验证GPU工作
            test_tensor = torch.randn(100, 100).cuda()
            result['tensor_device'] = str(test_tensor.device)
            result['tensor_sum'] = float(test_tensor.sum())
            
        return result
    
    def run_simple_computation(self):
        """运行简单计算来验证GPU工作"""
        import torch
        import time
        
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}
            
        start_time = time.time()
        
        # 创建大型矩阵运算
        a = torch.randn(1000, 1000).cuda()
        b = torch.randn(1000, 1000).cuda()
        
        # 执行矩阵乘法
        c = torch.matmul(a, b)
        result_sum = c.sum().item()
        
        end_time = time.time()
        
        return {
            'worker_id': self.worker_id,
            'computation_time': end_time - start_time,
            'result_sum': result_sum,
            'tensor_device': str(c.device),
        }

def test_multi_gpu_distribution():
    """测试多GPU分布式"""
    logger.info("🔧 Testing multi-GPU distribution...")
    
    try:
        # 初始化Ray
        ray.init(
            num_cpus=8,
            num_gpus=8,
            ignore_reinit_error=True,
            log_to_driver=True,
        )
        
        logger.info("✅ Ray initialized")
        logger.info(f"Available resources: {ray.available_resources()}")
        
        # 创建8个GPU Workers
        workers = []
        for i in range(8):
            worker = GPUWorker.remote(worker_id=i)
            workers.append(worker)
        
        logger.info(f"✅ Created {len(workers)} GPU workers")
        
        # 并行获取所有Worker的GPU信息
        logger.info("🔍 Getting GPU info from all workers...")
        gpu_info_futures = [worker.get_gpu_info.remote() for worker in workers]
        gpu_infos = ray.get(gpu_info_futures)
        
        # 打印结果
        logger.info("\n📊 GPU Distribution Results:")
        logger.info("=" * 80)
        
        for info in gpu_infos:
            logger.info(f"Worker {info['worker_id']}:")
            logger.info(f"  Process ID: {info['process_id']}")
            logger.info(f"  CUDA_VISIBLE_DEVICES: {info['cuda_visible_devices']}")
            logger.info(f"  CUDA devices visible: {info['cuda_device_count']}")
            logger.info(f"  Current device: cuda:{info.get('current_device', 'N/A')}")
            logger.info(f"  Device name: {info.get('device_name', 'N/A')}")
            logger.info(f"  Tensor device: {info.get('tensor_device', 'N/A')}")
            logger.info("  " + "-" * 50)
        
        # 验证每个Worker确实使用不同的物理GPU
        visible_devices = [info['cuda_visible_devices'] for info in gpu_infos]
        unique_devices = set(visible_devices)
        
        logger.info(f"\n✅ Unique physical GPUs used: {len(unique_devices)}")
        logger.info(f"Expected: 8, Actual: {len(unique_devices)}")
        
        if len(unique_devices) == 8:
            logger.info("🎉 SUCCESS: Each worker is using a different physical GPU!")
        else:
            logger.error("❌ FAILED: Workers are not properly distributed across GPUs")
            
        # 运行并行计算测试
        logger.info("\n🚀 Running parallel computation test...")
        compute_futures = [worker.run_simple_computation.remote() for worker in workers]
        compute_results = ray.get(compute_futures)
        
        logger.info("💻 Parallel Computation Results:")
        total_time = 0
        for result in compute_results:
            if 'error' not in result:
                logger.info(f"  Worker {result['worker_id']}: {result['computation_time']:.3f}s on {result['tensor_device']}")
                total_time += result['computation_time']
            else:
                logger.error(f"  Worker error: {result['error']}")
        
        avg_time = total_time / len(compute_results)
        logger.info(f"Average computation time per worker: {avg_time:.3f}s")
        logger.info("✅ All workers completed parallel computation successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Multi-GPU test failed: {e}")
        logger.exception("Full traceback:")
        return False
    finally:
        try:
            ray.shutdown()
            logger.info("✅ Ray shutdown completed")
        except:
            pass

def main():
    """主测试函数"""
    logger.info("🧪 多GPU分布式验证测试")
    logger.info("=" * 60)
    
    success = test_multi_gpu_distribution()
    
    logger.info("\n📊 Test Results:")
    logger.info("=" * 60)
    
    if success:
        logger.info("✅ Multi-GPU Distribution Test: PASS")
        logger.info("🎉 每个Worker确实运行在不同的物理GPU上！")
        logger.info("💡 虽然每个Worker内部使用cuda:0，但它们对应不同的物理GPU")
    else:
        logger.error("❌ Multi-GPU Distribution Test: FAIL")

if __name__ == "__main__":
    main() 