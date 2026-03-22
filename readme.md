# Generate 4D Heart

## 环境配置

此项目使用 [pixi](https://pixi.sh/) 配置环境

使用 `pixi install --frozen` 从 `pixi.lock` 安装环境。该方法确保环境的一致性，避免因依赖版本变化引起的问题。

如果安装失败，可能是由于某些依赖包在当前环境中无法安装。可以尝试以下步骤：
+ 使用 `pixi install` 重新安装，这将允许pixi根据当前环境自动调整依赖版本。
+ 如果仍然无法安装，检查错误日志，确定是哪个包导致安装失败，并修改 `pixi.toml` 中对应包的版本

安装完成后，可以使用 `pixi shell` 进入环境，或者使用 `pixi run <command>` 直接运行命令，例如 `pixi run python generate_4d_dvf.py`。

## 使用

目前项目仍在开发中，尚未实现完整稳定的接口，可通过查看`scripts/` 和 `tests/`目录下的脚本和测试用例了解项目的使用方法和功能实现。

## 主要架构

### 心脏运动生成
目前项目中包含三种心脏运动生成方式（主要在data_reader部分有区分）
1. 体模数据提取SSM（已提前得到）-> 运动心腔标签 -> 深度配准模型（shape morpher）-> 多帧速度场 -> 多帧形变场 -> 运动图像、标签 -> 旋转DSA
    - 生成形变场部分对应 `generate_4d_dvf.py` 与 `gnerate_4d_heart/moving_dvf`，输入为SSM参数和心腔标签，输出为多帧速度场。因为速度场空间较大，需要提前生成并保存。
    - TODO 生成形变场部分的测试。
    - 生成旋转DSA部分对应`generate_4d_heart/rotate_dsa/data_reader/volume_dvf_reader.py`及相关测试，以及`scripts/batch_dvf_to_dsa.py`，输入为图像、标签和上一步生成的速度场文件，输出为旋转DSA图像和标签
2. 体模数据提取SSM + 心腔标签 -> RBF插值运动场 -> 形变场 -> 运动图像、标签 -> 旋转DSA
    - 对应`generate_4d_heart/rotate_dsa/data_reader/rbf_reader.py`及相关测试,以及`scripts/batch_rbf_to_dsa.py`，输入入为图像和标签，输出为旋转DSA
3. 使用静态数据、运动体模CCTA直接生成旋转DSA
    - 对应其他reader文件

### DRR投影
目前仅实现了一种投影方式，基于[DiffDRR](https://github.com/eigenvivek/DiffDRR)，底层使用pytorch进行光线步进投影。对应`generate_4d_heart/rotate_dsa/rotate_drr/torch_drr.py`

### 造影剂模拟

TODO 存在问题，需要修改