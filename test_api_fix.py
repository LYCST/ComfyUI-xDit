#!/usr/bin/env python3
"""
测试xFuser API修复
=================

验证xFuserFluxPipeline.from_pretrained的API调用是否正确
"""

import os
import sys
import torch
import logging

# 设置环境
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "custom_nodes", "comfyui_xdit_multigpu"))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_xfuser_api():
    """测试xFuser API调用"""
    try:
        # 尝试导入xDiT
        try:
            import xfuser
            from xfuser import xFuserFluxPipeline
            from xfuser.config.config import EngineConfig
            logger.info("✅ xDiT imports successful")
        except ImportError as e:
            logger.error(f"❌ xDiT import failed: {e}")
            return False
            
        # 测试EngineConfig创建
        try:
            engine_config = EngineConfig(
                model_name="/fake/path/test",
                strategy="Hybrid"
            )
            logger.info("✅ EngineConfig creation successful")
        except Exception as e:
            logger.error(f"❌ EngineConfig creation failed: {e}")
            return False
            
        # 测试API签名（不实际加载模型）
        try:
            # 检查xFuserFluxPipeline.from_pretrained的签名
            import inspect
            sig = inspect.signature(xFuserFluxPipeline.from_pretrained)
            logger.info(f"API signature: {sig}")
            
            # 检查是否有engine_config参数
            if 'engine_config' in sig.parameters:
                logger.info("✅ engine_config parameter found in API")
                return True
            else:
                logger.error("❌ engine_config parameter not found in API")
                return False
                
        except Exception as e:
            logger.error(f"❌ API signature check failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.exception("Full traceback:")
        return False

def main():
    """主测试函数"""
    logger.info("="*40)
    logger.info("xFuser API修复测试")
    logger.info("="*40)
    
    success = test_xfuser_api()
    
    logger.info("="*40)
    logger.info(f"测试结果: {'✅ API修复成功' if success else '❌ 仍有问题'}")
    logger.info("="*40)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 