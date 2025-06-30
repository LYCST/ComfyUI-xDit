#!/usr/bin/env python3
"""
测试xDiT多GPU功能的脚本
"""
import os
import sys
import logging

# 设置环境变量
os.environ['XDIT_DEBUG'] = '1'  # 开启调试模式

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_single_gpu():
    """测试单GPU模式"""
    logger.info("=== 测试单GPU模式 ===")
    # TODO: 添加单GPU测试代码
    
def test_multi_gpu():
    """测试多GPU模式"""
    logger.info("=== 测试多GPU模式 ===")
    # TODO: 添加多GPU测试代码

if __name__ == "__main__":
    # 先测试单GPU
    test_single_gpu()
    
    # 再测试多GPU
    # test_multi_gpu()