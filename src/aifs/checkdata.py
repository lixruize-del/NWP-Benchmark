import numpy as np
import os
import datetime

# 指定你要检查的文件路径
# (根据之前的日志，这是生成的文件位置)
FILE_PATH = "assets/data/processed_aifs/init_20230101_12.npz"

def inspect_data():
    if not os.path.exists(FILE_PATH):
        print(f"❌ 文件不存在: {FILE_PATH}")
        print("请先运行 src/aifs/prepare_new.py")
        return

    print(f"🔍 正在检查文件: {FILE_PATH}")
    
    try:
        # 加载 npz
        data = np.load(FILE_PATH, allow_pickle=True)
        
        # 1. 检查日期
        if 'date' in data:
            date_val = data['date']
            print(f"📅 数据日期: {date_val}")
        else:
            print("⚠️ 警告: 'date' 字段缺失")

        # 2. 获取所有变量名
        keys = list(data.keys())
        # 过滤掉 'date'，只看物理变量
        vars = [k for k in keys if k != 'date']
        print(f"📦 包含变量总数: {len(vars)}")
        
        # 3. 关键变量检查清单
        # AIFS 推理通常必需的变量
        critical_vars = ['tp', 'z', 'lsm', 'sdor', 'slor', '2t', 'msl', '10u', '10v']
        
        print("\n--- 关键变量检查 ---")
        for v in critical_vars:
            if v in data:
                shape = data[v].shape
                # 检查是否有 NaN
                has_nan = np.isnan(data[v]).any()
                # 检查是否全 0
                is_all_zero = np.all(data[v] == 0)
                
                status = "✅ 正常"
                if has_nan: status = "❌ 包含 NaN"
                if is_all_zero: status = "⚠️ 全为 0 (可能是填充的)"
                
                print(f"{v:<5} | Shape: {str(shape):<15} | {status}")
            else:
                print(f"{v:<5} | ❌ 缺失 (Missing)")

        # 4. 打印前 10 个存在的变量名 (用于确认命名风格)
        print("\n--- 变量名示例 (前10个) ---")
        print(vars[:10])

        # 5. 维度一致性检查
        # 假设所有变量的第一维应该是 2 (T-6, T0)，第二维是网格点数 (N320约 542080)
        print("\n--- 维度一致性检查 ---")
        shapes = {}
        for v in vars:
            s = data[v].shape
            if s not in shapes:
                shapes[s] = []
            shapes[s].append(v)
        
        for s, v_list in shapes.items():
            print(f"Shape {s}: {len(v_list)} 个变量")
            # 如果数量少，打印出来看看是哪些
            if len(v_list) < 5:
                print(f"  -> {v_list}")

    except Exception as e:
        print(f"❌ 读取错误: {e}")

if __name__ == "__main__":
    inspect_data()