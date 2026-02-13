"""
批量生成 Plan A/B/C 的对比图
- 预测精度分析（微观视图 + 相关性）
- GOE谱分析（能级间距统计 + KS距离）
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sympy import primerange
import os
from pathlib import Path

# ==================== 配置区 ====================
WEIGHT_DIR = r"D:\pythonstudy\python_task\权重分析"
OUTPUT_DIR = r"D:\pythonstudy\python_task\权重分析\论文图片"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 实验配置
EXPERIMENTS = {
    "Plan_A": {
        "weight_file": "Plan A.pt",
        "learnable_embedding": True,
        "label": "Plan A (Learnable + Large Batch)",
        "color": "#E74C3C"  # 红色
    },
    "Plan_B": {
        "weight_file": "Plan B.pt",
        "learnable_embedding": True,
        "label": "Plan B (Learnable + Small Batch)",
        "color": "#3498DB"  # 蓝色
    },
    "Plan_C": {
        "weight_file": "Plan C.pt",
        "learnable_embedding": False,  # 正弦波编码
        "label": "Plan C (Sinusoidal + Large Batch)",
        "color": "#2ECC71"  # 绿色
    }
}

# 模型参数
D_MODEL = 256
N_LAYERS = 6
N_HEADS = 8
DROPOUT = 0.1
NUM_PRIMES = 1000000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ 使用设备: {device}\n")

# ==================== 模型定义 ====================
class RiemannEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000000, learnable=True):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.embedding = nn.Embedding(max_len, d_model)
        else:
            # 正弦波编码
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

    def forward(self, x):
        if self.learnable:
            return self.embedding(x)
        else:
            return self.pe[x]

class PrimeGapPredictor(nn.Module):
    def __init__(self, learnable_embedding=True):
        super().__init__()
        self.riemann_embedding = RiemannEmbedding(D_MODEL, NUM_PRIMES, learnable=learnable_embedding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_MODEL*4,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.output = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 2, D_MODEL // 4),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 4, 1)
        )

    def forward(self, x):
        embedded = self.riemann_embedding(x).unsqueeze(1)
        transformed = self.transformer(embedded)
        return self.output(transformed.squeeze(1))

# ==================== 分析函数 ====================

def load_model(weight_path, learnable_embedding):
    """加载模型"""
    model = PrimeGapPredictor(learnable_embedding=learnable_embedding).to(device)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model

def analyze_prediction_accuracy(model, plan_name, color):
    """预测精度分析（完全照抄源代码）"""
    print(f"  🔍 分析预测精度...")

    # 生成全部素数并计算全局统计量
    all_primes = list(primerange(1, 18500000))[:NUM_PRIMES]
    all_gaps = np.diff(all_primes)

    global_mean = np.mean(all_gaps)
    global_std = np.std(all_gaps)

    # 取最后1000个gap
    total_gaps = len(all_gaps)
    target_len = 1000
    start_idx = total_gaps - target_len
    end_idx = total_gaps
    target_gaps = all_gaps[start_idx:end_idx]

    # 预测
    indices = torch.arange(start_idx, end_idx, device=device)
    with torch.no_grad():
        preds_norm = model(indices).squeeze().cpu().numpy()

    # 还原归一化
    preds_real = (preds_norm * global_std) + global_mean

    # 计算误差
    mae = np.mean(np.abs(preds_real - target_gaps))

    # 绘图
    plt.figure(figsize=(18, 6))

    # 左图：微观视图 (前200个)
    plt.subplot(1, 2, 1)
    plt.plot(target_gaps[:200], color='black', alpha=0.6, label='Real Truth', linewidth=2)
    plt.plot(preds_real[:200], color='red', alpha=0.8, linestyle='--', label='AI Prediction', linewidth=1.5)
    plt.title(f'Micro View: First 200 of Last 1000 Gaps')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 右图：整体相关性
    plt.subplot(1, 2, 2)
    plt.scatter(target_gaps, preds_real, alpha=0.5, s=10, c='blue')
    min_v = min(target_gaps.min(), preds_real.min())
    max_v = max(target_gaps.max(), preds_real.max())
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', label='Perfect Fit')
    plt.title(f'Correlation (MAE={mae:.4f})')
    plt.xlabel('Real Gap')
    plt.ylabel('Predicted Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'{plan_name}_prediction.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✅ MAE = {mae:.4f}, 图片已保存")
    return mae

def analyze_goe_spectrum(model, plan_name, color):
    """GOE谱分析（包含KS距离和直方图拟合MAE）"""
    print(f"  🔬 分析GOE谱...")

    # 提取权重
    weights = []
    for name, param in model.named_parameters():
        if 'in_proj_weight' in name:
            weights.append(param.detach().cpu().numpy())

    if not weights:
        for name, param in model.named_parameters():
            if len(param.shape)==2 and param.shape[0]==param.shape[1]:
                weights.append(param.detach().cpu().numpy())

    if not weights:
        print("    ⚠️ 未找到可用权重矩阵")
        return None

    # 拼接并截取（严格按论文方法）
    W_huge = np.concatenate(weights, axis=0)
    n = min(2048, W_huge.shape[0], W_huge.shape[1])
    W = W_huge[:n, :n]

    # 厄米化
    H = (W + W.T) / 2
    eigvals = np.linalg.eigvalsh(H)

    # 计算能级间距
    eigvals = np.sort(eigvals)
    limit_low = int(n * 0.15)
    limit_high = int(n * 0.85)
    eigvals = eigvals[limit_low : limit_high]
    spacings = np.diff(eigvals)
    s = spacings / np.mean(spacings)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(s, bins=70, density=True, alpha=0.65, color=color, edgecolor='black', label=f'AI Weights ({plan_name})')

    x = np.linspace(0, 4, 300)
    p_goe = (np.pi / 2) * x * np.exp(-np.pi * x**2 / 4)
    p_poisson = np.exp(-x)
    ax.plot(x, p_goe, 'r-', linewidth=3, label='GOE (Time-Reversal Symmetric Chaos)')
    ax.plot(x, p_poisson, 'g--', linewidth=3, label='Poisson (Random)')

    ax.set_title(f'{plan_name}: Level Spacing Statistics', fontsize=14, fontweight='bold')
    ax.set_xlabel('Normalized Spacing (s)', fontsize=12)
    ax.set_ylabel('Probability Density P(s)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3.5)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'{plan_name}_goe_spectrum.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --- 计算距离指标 ---
    # 1. 直方图拟合 MAE（原来的指标，现重命名）
    def calc_pdf_mae(data, pdf_func):
        y_hist, bins = np.histogram(data, bins=100, density=True, range=(0, 3))
        centers = (bins[:-1] + bins[1:]) / 2
        return np.mean(np.abs(y_hist - pdf_func(centers)))

    mae_goe = calc_pdf_mae(s, lambda x: (np.pi / 2) * x * np.exp(-np.pi * x ** 2 / 4))
    mae_poisson = calc_pdf_mae(s, lambda x: np.exp(-x))

    # 2. 真实的 Kolmogorov-Smirnov 距离
    def goe_cdf(s):
        return 1 - np.exp(-np.pi * s ** 2 / 4)

    def poisson_cdf(s):
        return 1 - np.exp(-s)

    s_valid = s[s <= 4]  # 排除过大值，不影响 KS 统计量
    ks_goe = stats.kstest(s_valid, goe_cdf).statistic
    ks_poisson = stats.kstest(s_valid, poisson_cdf).statistic

    # 基于 KS 距离的判决（哪个更小）
    verdict_ks = "GOE" if ks_goe < ks_poisson else "Poisson"

    print(f"    ✅ KS distance (vs GOE)     = {ks_goe:.4f}")
    print(f"       KS distance (vs Poisson) = {ks_poisson:.4f}")
    print(f"       MAE fit (vs GOE)         = {mae_goe:.4f} (non‑KS, legacy metric)")
    print(f"       MAE fit (vs Poisson)     = {mae_poisson:.4f}")

    return {
        "verdict": verdict_ks,  # 改用 KS 判决
        "mae_goe": mae_goe,
        "mae_poisson": mae_poisson,
        "ks_goe": ks_goe,
        "ks_poisson": ks_poisson
    }
# ==================== 主流程 ====================

def main():
    print("="*60)
    print("🚀 批量生成 Plan A/B/C 对比图")
    print("="*60)

    results = {}

    for plan_name, config in EXPERIMENTS.items():
        print(f"\n📦 处理 {plan_name}...")

        weight_path = os.path.join(WEIGHT_DIR, config["weight_file"])
        if not os.path.exists(weight_path):
            print(f"  ⚠️ 权重文件不存在: {weight_path}")
            continue

        print(f"  ⏳ 加载权重: {config['weight_file']}")
        model = load_model(weight_path, config["learnable_embedding"])

        mae = analyze_prediction_accuracy(model, plan_name, config["color"])
        goe_result = analyze_goe_spectrum(model, plan_name, config["color"])

        results[plan_name] = {
            "mae_pred": mae,
            "goe": goe_result
        }

    # 生成汇总报告
    print("\n" + "=" * 60)
    print("📊 实验汇总")
    print("=" * 60)
    for plan_name, result in results.items():
        print(f"\n【{plan_name}】")
        print(f"  预测精度: MAE = {result['mae_pred']:.4f}")
        if result['goe']:
            print(f"  GOE谱分析:")
            print(f"    - KS(GOE)     = {result['goe']['ks_goe']:.4f}")
            print(f"    - KS(Poisson) = {result['goe']['ks_poisson']:.4f}")
            print(f"    - MAE(GOE)     = {result['goe']['mae_goe']:.4f} (legacy)")
            print(f"    - MAE(Poisson) = {result['goe']['mae_poisson']:.4f} (legacy)")
            print(f"    判决 (基于KS): {result['goe']['verdict']}")

if __name__ == "__main__":
    main()
