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

## 转化为 nerfstudio 和 instant-ngp 支持的格式

使用 `scripts/dataset_to_colmap.py` 将生成的旋转DSA图像和标签转化为colmap格式，输入为旋转DSA图像和标签文件夹，输出为colmap格式的文件夹。

