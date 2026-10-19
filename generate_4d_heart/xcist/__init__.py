"""XCIST(gecatsim) 投影后端 spike 模块。

本包把现有 RotateDSA/DiffDRR 的 C-arm 几何协议映射到 GE 开源的
X-ray CT simulator (gecatsim)，输出多能谱 + 平板探测器的投影图，
用于和 DiffDRR(单能) 对比评估物理真实性。

注意：gecatsim 不可微分、输入是材料密度体(非 μ/HU)，因此本包是
独立旁路，不实现 rotate_drr.RotateDRR 协议（分层加和的线积分架构
无法平移到 xcist 的单卷多能投影）。
"""
