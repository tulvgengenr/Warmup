import torch
import torch.nn as nn

# 创建输入张量
in_t = torch.rand(size=(3, 2), requires_grad=True)
print("输入张量：", in_t)

# 创建 LayerNorm 模型
torch_layerNorm = nn.LayerNorm(normalized_shape=(2), elementwise_affine=False, eps=1e-5)

# 前向传播
out_torch = torch_layerNorm(in_t)
print("前向传播输出：", out_torch)

# 创建损失函数
loss_fn = torch.nn.MSELoss()

# 创建目标输出
target = torch.ones_like(out_torch)

# 计算损失
loss = loss_fn(out_torch, target)
print("损失值：", loss.item())

# 反向传播
loss.backward()

# 打印梯度
# print("权重梯度：", torch_layerNorm.weight.grad)
# print("偏置梯度：", torch_layerNorm.bias.grad)
print("输入张量的梯度：", in_t.grad)