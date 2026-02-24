import sys
import os
import torch
import jax
import onnxruntime

print("="*60)
print("🛠️ NWPBench 统一环境全栈自检")
print("="*60)

# 1. 检查 PyTorch & CUDA (H20 适配性)
print(f"\n[1] PyTorch Core:")
print(f"   - Version: {torch.__version__}")
print(f"   - CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   - GPU: {torch.cuda.get_device_name(0)}")
    print(f"   - CUDA Version: {torch.version.cuda}")
    print(f"   - CuDNN Version: {torch.backends.cudnn.version()}")

# 2. 检查 JAX & CUDA (NeuralGCM/GraphCast)
print(f"\n[2] JAX Core:")
try:
    print(f"   - Version: {jax.__version__}")
    print(f"   - Devices: {jax.devices()}")
    # 简单的计算测试
    x = jax.numpy.array([1.0, 2.0])
    y = x * 2
    print(f"   - JAX Calculation Test: Passed ({y})")
except Exception as e:
    print(f"   - ❌ Failed: {e}")

# 3. 检查模型特定库
print(f"\n[3] Model Dependencies Check:")

# AIFS
try:
    import anemoi.inference
    import anemoi.models
    print("   - ✅ AIFS (anemoi): Installed")
except ImportError as e:
    print(f"   - ❌ AIFS Missing: {e}")

# Aurora
try:
    import aurora
    print("   - ✅ Aurora (microsoft-aurora): Installed")
except ImportError as e:
    print(f"   - ❌ Aurora Missing: {e}")

# NeuralGCM
try:
    import neuralgcm
    import dinosaur
    print("   - ✅ NeuralGCM: Installed")
except ImportError as e:
    print(f"   - ❌ NeuralGCM Missing: {e}")

# GraphCast
try:
    import graphcast
    import haiku
    import jraph
    print("   - ✅ GraphCast (DeepMind): Installed")
except ImportError as e:
    print(f"   - ❌ GraphCast Missing: {e}")

# Pangu (ONNX)
try:
    providers = onnxruntime.get_available_providers()
    print(f"   - ✅ Pangu (ONNX): Installed (Providers: {providers})")
    if 'CUDAExecutionProvider' not in providers:
        print("     ⚠️ Warning: ONNX Runtime cannot find CUDA!")
except ImportError as e:
    print(f"   - ❌ Pangu Missing: {e}")

# Stormer (Accelerators)
try:
    import xformers
    import xformers.ops
    print(f"   - ✅ Stormer (xformers): Installed (v{xformers.__version__})")
except ImportError as e:
    print(f"   - ❌ Stormer (xformers) Missing: {e}")

# 4. 检查通用科学计算库
print(f"\n[4] Data Libs Check:")
try:
    import numpy
    import scipy
    import cfgrib
    print(f"   - Numpy: {numpy.__version__} (< 2.0 required?)")
    print(f"   - Scipy: {scipy.__version__} (<= 1.12 recommended for NeuralGCM)")
    print("   - cfgrib: Installed")
except ImportError as e:
    print(f"   - ❌ Data Libs Missing: {e}")

print("\n" + "="*60)