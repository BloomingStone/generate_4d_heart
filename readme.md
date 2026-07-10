# Generate 4D Heart

## 环境配置

此项目使用 [pixi](https://pixi.sh/) 配置环境

使用 `pixi install --frozen` 从 `pixi.lock` 安装环境。该方法确保环境的一致性，避免因依赖版本变化引起的问题。

如果安装失败，可能是由于某些依赖包在当前环境中无法安装。可以尝试以下步骤：
+ 使用 `pixi install` 重新安装，这将允许pixi根据当前环境自动调整依赖版本。
+ 如果仍然无法安装，检查错误日志，确定是哪个包导致安装失败，并修改 `pixi.toml` 中对应包的版本

安装完成后，可以使用 `pixi shell` 进入环境，或者使用 `pixi run <command>` 直接运行命令，例如 `pixi run python generate_4d_dvf.py`。

## 使用

目前项目仍在开发中，尚未实现完整稳定的接口，推荐clone后参照通过查看`scripts/` 和 `tests/`目录下的脚本和测试用例了解项目的使用方法和功能，并实现自己的数据处理脚本。

可以使用`pytest`运行`test/`目录下的测试用例，`test/test_data` 可以在 https://drive.google.com/file/d/1msaPhdpexfv6hyR9BKpY_W6dSuMaYh_q/view?usp=sharing 下载

### scripts 与 test

- **scripts**: 用于批量化处理、数据生成与可视化验证的实用脚本集合。包含常见的批处理流水线（例如批量从 DVF 生成旋转 DSA、RBF 批量生成、将数据转为 COLMAP 格式等），以及若干用于离线可视化、部署或 notebook 演示的入口脚本。

- **test**: 存放单元/集成测试和使用样例的目录，既用于本地自动化测试，也作为调试和功能示例参考。测试中展示了期望的模块接口和调用姿势（例如 `reader.get_data(phase, coronary_type)`、`simulator.preprocess()` + `simulate()` / `simulate_with_time()`、以及 `RotateDSA(reader=..., drr=...)` 的初始化方式），可以作为脚本改写和新功能接入时的参考。

## 主要架构

### 心脏运动生成
目前项目中包含三种心脏运动生成方式（主要在data_reader部分有区分）
1. 体模数据提取SSM（已提前得到）-> 运动心腔标签 -> 深度配准模型（shape morpher）-> 多帧速度场 -> 多帧形变场 -> 运动图像、标签 -> 旋转DSA
    - 生成形变场部分对应 `generate_4d_dvf.py` 与 `gnerate_4d_heart/moving_dvf`，输入为SSM参数和心腔标签，输出为多帧速度场。因为速度场空间较大，需要提前生成并保存。
    - TODO 生成形变场部分的测试。
    - 生成旋转DSA部分对应`generate_4d_heart/rotate_dsa/data_reader/volume_dvf_reader.py`及相关测试，以及`scripts/batch_dvf_to_dsa.py`，输入为图像、标签和上一步生成的速度场文件，输出为旋转DSA图像和标签
    - 该方法所得DVF在心脏处运动幅度较小，因此需要使用 `MovementEnhancer` 进行增强。但运动始终存在不平滑、不自然等情况。因此目前不退家使用
2. 体模数据提取SSM + 心腔标签 -> RBF插值运动场 -> 形变场 -> 运动图像、标签 -> 旋转DSA（**推荐使用**）
    - 对应`generate_4d_heart/rotate_dsa/data_reader/rbf_reader.py`及相关测试,以及`scripts/batch_rbf_to_dsa.py`，输入入为图像和标签，输出为旋转DSA
3. 使用静态数据、运动体模CCTA直接生成旋转DSA
    - 对应其他reader文件

### DRR投影
目前仅实现了一种投影方式，基于[DiffDRR](https://github.com/eigenvivek/DiffDRR)，底层使用pytorch进行光线步进投影。对应`generate_4d_heart/rotate_dsa/rotate_drr/torch_drr.py`

### 造影剂模拟

造影剂（contrast）模拟模块现已趋于稳定，主要代码位于：

- [generate_4d_heart/rotate_dsa/contrast_simulator](generate_4d_heart/rotate_dsa/contrast_simulator)

主要设计与使用要点：

- **接口约定**: 大多数模拟器提供 `preprocess(volume, cavity_label)` 返回基线衰减图；若支持时间变化则实现 `simulate_with_time(volume, cavity_label, coronary_label, time)`，否则实现 `simulate(volume, cavity_label, coronary_label)`；可通过 `contrast_change_over_time` 字段判断是否为时变模拟器。
  - TODO：给simulator重新取下名字，现在的不太准确。
  
- **常见实现**: `FlowContrast`（基于血流时间延迟与脉冲形状的时变模型）、`StaticIodineContrast`（基于标签进行固定吸收率变换）、`ThresholdIodineContrast` （基于阈值进行吸收率变换，主要依赖 `ori_volume` 的 voxel value 分层并结合 coronary label）、`IdentityContrast`（恒等模拟）等，位于该目录下的各个模块文件中。可以直接通过工厂函数或在脚本中实例化并传入 reader（例如 `RBFReader(..., contrast_simulator=StaticIodineContrast(),...)`）。
  
- **调用示例**: 在测试和 reader 中常见写法：先 `preprocess`（可选），然后根据是否为时变模拟器选择 `simulate` 或 `simulate_with_time`，例如测试中使用：

    - `preprocessed = simulator.preprocess(volume, cavity_label)`
    - `simulated = simulator.simulate(preprocessed, cavity_label, coronary_label)` 或
    - `simulated = simulator.simulate_with_time(preprocessed, cavity_label, coronary_label, time=0.5)`

参阅测试以获取更详细的调用样例。

### RotateDSA 输出说明

`RotateDSA.run_and_save(output_dir, coronary_type)` 运行完整的旋转DSA生成流程并将结果保存到指定目录，各输出文件说明如下：

#### 图像/视频文件

| 输出文件 | 说明 |
|---|---|
| `rotate_dsa_raw.tif` | 原始DRR密度投影，未经过后处理 |
| `rotate_dsa.gif` | 后处理后的可视化 GIF（uint8，经 `postprocess_drr` 处理，含gamma变换和灰度窗裁剪，最后默认转换为 uint8，值域 0–255） |
| `rotate_dsa_raw.gif` | 原始帧 GIF（float32 直接映射到灰度） |
| `rotate_dsa/` | 每帧后处理后的 PNG 图像（`000.png`, `001.png`, ...） |

#### 标签文件

| 输出文件 | 说明 |
|---|---|
| `label.gif` | 标签 GIF（值×255 后显示） |
| `label/` | 每帧标签的 PNG 图像 |
| `label.npz` | 标签的 NumPy 压缩存档（shape: `[T, H, W]`, dtype: uint8） |

#### 深度图文件

| 输出文件 | 说明 |
|---|---|
| `depth_map.gif` | 深度图 GIF（灰度，仅冠脉区域有有效值，背景为 0） |
| `depth_map.npz` | 深度图的 NumPy 压缩存档（shape: `[T, H, W]`, dtype: float32） |

#### 几何与元数据

| 输出文件 | 说明 |
|---|---|
| `rotate_dsa.json` | JSON 文件，包含冠脉类型、体积仿射矩阵、C 臂几何参数、旋转参数、逐帧角度/相位/时间信息、后处理元数据（`postprocess_meta`）以及 Phase 0 保存元数据（`phase_0_save_meta`） |
| Phase 0 数据 | 通过 `reader.get_phase_0_data().save()` 保存的冠脉 NIfTI 文件、网格（`.vtp`/`.ply`）、中心线（`.vtp`）等 |

## 转化为 nerfstudio 和 instant-ngp 支持的格式

使用 `scripts/dataset_to_colmap.py` 将生成的旋转DSA图像和标签转化为colmap格式，输入为旋转DSA图像和标签文件夹，输出为colmap格式的文件夹。


## 人造结构（Artifact）模拟

人造结构（导丝、标记点、支架等）通过 mesh 表示，由 `generate_4d_heart/rotate_dsa/artifacts/` 下的模块管理。

### Artifact 数据类

```python
@dataclass
class Artifact:
    mesh: pv.PolyData   # mesh 表示，世界坐标系
    mu: float           # 线衰减系数 (mm⁻¹)
    name: str           # 标识符
```

### 生成方式

- **导丝** (`artifacts/guide_wire.py`): 对冠脉标签骨架化 → BFS 寻路 → 平滑 → `pv.Tube` 生成管状 mesh
- **标记点** (`artifacts/marker.py`): 沿导丝中心线按物理间距采样 → `pv.Sphere` 生成球体 mesh（固定 2 个标记点）
- **外部输入**: 通过 `ExternalArtifactConfig` 加载外部 `.vtp`/`.ply` 文件

### ArtifactManager

配置驱动工厂，按序处理依赖（导丝 → 标记点 → 外部文件），在 RBFReader 初始化时调用，生成的 artifacts 存储在 `Coronary.artifacts` 中。

```python
config = ArtifactManagerConfig(
    guide_wire=GuideWireConfig(radius_mm=0.3),
    markers=MarkerConfig(distance_mm=8.0, radius_mm=0.4),
)
reader = RBFReader(
    ...,
    artifact_manager=ArtifactManager(config),
)
```

### 渲染

`Torch3DLabelRenderer.render_line_integral()` 使用 PyTorch3D rasterizer 计算 mesh 的 X 射线线积分贡献（`faces_per_pixel=2` → `path_length = back_depth - front_depth`），结果叠加在 DRR 的 line integral 上。

## 中心线长度保持约束（Centerline Length Constraint）

RBF 形变场将 SSM 控制点位移插值到全部空间点。导丝、支架等人造结构附着在冠脉表面，随 RBF 形变时会被非均匀拉伸或压缩。该项目在 `KDTreeRBF` 中实现了一阶中心线长度保持约束，从源头抑制这种非物理变形。

### 推导

设 $a_i \in \mathbb{R}^3$ 为冠脉中心线点（LCA 与 RCA），$c_j \in \mathbb{R}^3$ 为 SSM 控制点，$D_j \in \mathbb{R}^3$ 为控制点位移。RBF 插值使用归一化高斯核：

$$v_i = \sum_{j} w_{ij} D_j,\qquad w_{ij} = \frac{\exp(-\|a_i - c_j\|^2 / 2\sigma^2)}{\sum_{k} \exp(-\|a_i - c_k\|^2 / 2\sigma^2)}$$

相邻中心线点 $a_i, a_{i+1}$ 的边向量为 $e_i = a_{i+1} - a_i$。形变后该边的长度变化一阶展开为：

$$\|(a_{i+1} + v_{i+1}) - (a_i + v_i)\| - \|e_i\| \approx \hat{e}_i \cdot (v_{i+1} - v_i) + O(\|v\|^2)$$

即 $\Delta L \approx \hat{e} \cdot \Delta v$ ——边长的**一阶变化等于位移差在边单位方向上的投影**。如图所示：

```
      Δv₁ ──→           Δv₁
    ●───────●         ●───●
   aᵢ      aᵢ₊₁     aᵢ'  aᵢ₊₁'
    
    Δv ∥ ê → 拉伸/压缩
    Δv ⟂ ê → 弯曲（不影响长度）
```

令一阶项为零，得到线性约束：

$$e_i \cdot (v_{i+1} - v_i) = 0$$

代入 $v_i = \sum_j w_{ij} D_j$，记 $\Delta w_{ij} = w_{i+1,j} - w_{i,j}$：

$$\sum_{j} \Delta w_{ij}\ (e_i \cdot D_j) = 0$$

将所有 $K$ 条边的约束写成矩阵形式 $A c = 0$，其中 $c \in \mathbb{R}^{3M}$ 是堆叠的控制点位移向量（$M$ 个控制点，每点 3 坐标），$A \in \mathbb{R}^{K \times 3M}$ 的行构造为：

$$A_{i,\ 3j + d} = \Delta w_{ij} \cdot e_i^{(d)},\qquad d \in \{0,1,2\}$$

为找到最接近原始 SSM 位移 $c_0$ 且满足约束的 $c'$，求解带约束最小二乘：

$$\min_{c'} \|c' - c_0\|^2 \quad \text{s.t.} \quad A c' = 0$$

该问题有闭式解（Lagrange 乘子法）：

$$c'_{\text{raw}} = c_0 - A^T (A A^T + \lambda I)^{-1} A c_0$$

其中 $\lambda = 10^{-10}$ 是用于处理秩亏的正则化项。该投影在 `KDTreeRBF._constrain_centerline_lengths()` 中实现。

### 位移幅值恢复

投影会去除 $c_0$ 中沿 $A$ 行空间的分量（即导致中心线拉伸/收缩的成分），导致 $\|c'_{\text{raw}}\| < \|c_0\|$（实测约减少 14%）。由于 $A$ 是线性算子，均匀缩放不破坏约束：

$$A(\alpha \cdot c') = \alpha \cdot A c' = \alpha \cdot 0 = 0$$

因此可以在投影后做一步 rescale 恢复原始位移强度：

$$c' = c'_{\text{raw}} \cdot \frac{\|c_0\|}{\|c'_{\text{raw}}\|}$$

这样得到的 $c'$ 同时满足：
- $A c' = 0$（中心线长度不变）
- $\|c'\| = \|c_0\|$（位移强度不变）

### 使用

默认开启，无需额外配置。可通过 `RBFReader` 的 `centerline_constraint_enabled` 字段控制：

```python
# 开启（默认）
reader = RBFReader(..., centerline_constraint_enabled=True)

# 关闭（使用原始 SSM 形变场）
reader = RBFReader(..., centerline_constraint_enabled=False)
```

初始化时输出约束统计：

```
KDTreeRBF: constraining 424 centreline edge-lengths across 500 control points for 20 phases
KDTreeRBF: mean constraint residual: 834.8875 → 0.000035 (float32),
           displacement norm ratio (pre-rescale): 0.8618 [0.8467, 0.8783] → rescaled to 1.0
```

- **约束残差**从 $834.89$ 降至 $3.5 \times 10^{-5}$（机器精度级别），一阶长度约束被精确满足。
- **位移幅值**在投影后约降为原来的 $86\%$（$14\%$ 的成分导致中心线拉伸/收缩），经 rescale 恢复至 $100\%$。

### 限制

- 该约束仅维持**一阶**长度不变。对于大形变，高阶（弯曲）项仍可能引入微小长度变化。
- 约束仅作用于冠脉中心线（LCA + RCA），不直接约束心肌等其他组织。心房、心室的收缩形变保持不变。
- 降采样参数 `centerline_downsample=3` 在精度与计算代价间取得平衡；若中心线过密或过疏，可调整。

### TODOs

+ [ ] **其他 reader 适配 artifact**: 目前仅在 `RBFReader` 中实现了 artifact 的生成与形变。`VolumesReader`、`VolumeDVFReader`、`StaticVolumeReader`、`StaticLabelReader` 等尚未接入 `artifact_manager` 参数，且 `get_data()` 中未对 artifact mesh 施加运动形变。
+ [ ] **外部输入 artifact 测试**: `ExternalArtifactConfig` 已定义，但尚未编写加载外部 `.vtp`/`.ply` 文件的单元测试和集成测试。
+ [ ] 增加对支架的模拟

