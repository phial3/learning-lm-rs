import torch
from transformers import LlamaConfig, LlamaModel
import numpy as np
import json
import os

# 根据您的Rust参数创建匹配的配置
with open("./models/story/config.json", "r") as f:
    config_dict = json.load(f)

# 使用与您的Rust代码完全匹配的参数
vocab_size = 2048
n_layers = 2
n_q_h = 8  # 查询头数量
n_kv_h = 4  # 键值头数量
hidden_size = 128
dqkv = 16  # 每个头的维度
di = 384  # 中间层大小

# 创建测试数据 - 匹配您的模型尺寸
batch_size = 1
seq_len = 8  # 输入序列长度
total_seq_len = 16  # 总序列长度(可能包括缓存的键值)

# 生成随机输入
input_data = torch.randn(batch_size, seq_len, hidden_size)

# 创建Q, K, V投影的模拟数据
# 对应于Rust代码中的函数参数格式
q_data = torch.randn(seq_len, n_q_h * dqkv)  # (seq, n_q_h * dqkv)
k_data = torch.randn(total_seq_len, n_kv_h * dqkv)  # (total_seq, n_kv_h * dqkv)
v_data = torch.randn(total_seq_len, n_kv_h * dqkv)  # (total_seq, n_kv_h * dqkv)

# 重新整形以便于PyTorch计算
# 将(seq, n_q_h * dqkv)转换为(batch, n_heads, seq, head_dim)格式
q_reshaped = q_data.view(seq_len, n_q_h, dqkv).unsqueeze(0)  # [1, seq, n_q_h, dqkv]
q_reshaped = q_reshaped.permute(0, 2, 1, 3)  # [1, n_q_h, seq, dqkv]

# 将(total_seq, n_kv_h * dqkv)转换为(batch, n_kv_h, total_seq, head_dim)格式
k_reshaped = k_data.view(total_seq_len, n_kv_h, dqkv).unsqueeze(0)  # [1, total_seq, n_kv_h, dqkv]
k_reshaped = k_reshaped.permute(0, 2, 1, 3)  # [1, n_kv_h, total_seq, dqkv]

v_reshaped = v_data.view(total_seq_len, n_kv_h, dqkv).unsqueeze(0)  # [1, total_seq, n_kv_h, dqkv]
v_reshaped = v_reshaped.permute(0, 2, 1, 3)  # [1, n_kv_h, total_seq, dqkv]

# 计算注意力分数 - 需要为GQA(分组查询注意力)做调整
attn_scores = torch.zeros(batch_size, n_kv_h, n_q_h // n_kv_h, seq_len, total_seq_len)

# 计算注意力分数
for kv_head in range(n_kv_h):
    # 对于每个KV头，计算相应的查询组
    group_size = n_q_h // n_kv_h
    start_q_head = kv_head * group_size
    end_q_head = (kv_head + 1) * group_size
    
    for group in range(group_size):
        q_head = start_q_head + group
        # Q[batch, q_head, seq, dqkv] @ K[batch, kv_head, total_seq, dqkv].transpose(-1, -2)
        scores = torch.matmul(
            q_reshaped[:, q_head:q_head+1], 
            k_reshaped[:, kv_head:kv_head+1].transpose(-1, -2)
        ) / (dqkv ** 0.5)
        
        # 将分数存储到正确的位置
        attn_scores[:, kv_head, group] = scores.squeeze(1)  # 移除多余的头维度

# 将形状调整为与您的Rust代码匹配的格式
attn_scores_reshaped = attn_scores.view(n_kv_h, n_q_h // n_kv_h, seq_len, total_seq_len)

# 应用softmax
attn_weights = torch.nn.functional.softmax(attn_scores_reshaped, dim=-1)

# 初始化结果变量
attn_output = torch.zeros(seq_len, n_q_h * dqkv)

# 计算注意力输出
for kv_head in range(n_kv_h):
    for group in range(n_q_h // n_kv_h):
        for seq in range(seq_len):
            for dim in range(dqkv):
                output_val = 0.0
                for total_seq in range(total_seq_len):
                    # 获取注意力权重
                    weight = attn_weights[kv_head, group, seq, total_seq].item()
                    # 获取对应的值向量元素
                    v_val = v_data[total_seq, kv_head * dqkv + dim].item()
                    output_val += weight * v_val
                
                # 将结果存储到正确的位置
                attn_output[seq, (kv_head * (n_q_h // n_kv_h) + group) * dqkv + dim] = output_val

# 保存用于与Rust比较的张量
np.save("q_tensor0.npy", q_data.numpy())
np.save("k_tensor0.npy", k_data.numpy())
np.save("v_tensor0.npy", v_data.numpy())
np.save("attn_scores_tensor0.npy", attn_scores_reshaped.numpy())
np.save("attn_weights_tensor0.npy", attn_weights.numpy())
np.save("attn_output_tensor0.npy", attn_output.numpy())

print("输入形状:")
print(f"q shape: {q_data.shape}")  # 应该是 [seq_len, n_q_h * dqkv]
print(f"k shape: {k_data.shape}")  # 应该是 [total_seq_len, n_kv_h * dqkv]
print(f"v shape: {v_data.shape}")  # 应该是 [total_seq_len, n_kv_h * dqkv]

print("\n中间结果形状:")
print(f"attention scores shape: {attn_scores_reshaped.shape}")  # 应该是 [n_kv_h, n_q_h/n_kv_h, seq_len, total_seq_len]
print(f"attention weights shape: {attn_weights.shape}")  # 应该是 [n_kv_h, n_q_h/n_kv_h, seq_len, total_seq_len]

print("\n输出形状:")
print(f"attention output shape: {attn_output.shape}")  # 应该是 [seq_len, n_q_h * dqkv]

# 打印几个样本值以便调试
print("\n样本值:")
print(f"First q value: {q_data[0, 0]}")
print(f"First k value: {k_data[0, 0]}")
print(f"Sample attention score: {attn_scores_reshaped[0, 0, 0, 0]}")
print(f"Sample attention weight: {attn_weights[0, 0, 0, 0]}")
print(f"Sample output value: {attn_output[0, 0]}")