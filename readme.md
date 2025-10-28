TODO 将分散的CLI入口统一整合到main.py中，或是增加 scripts 演示脚本
TODO scripts 演示脚本需要增加全量 旋转 DSA 处理逻辑

TODO 测试使用无造影剂图像生成DSA的效果（Phantom 与 CTA2CT 图像）
TODO 测试使用高分辨率XCAT生成DSA

DONE 需要完成冠脉运动增强处的逻辑
    - LATER 目前phantom体模的一个缺点是没有心脏整体的收缩，或许可以用心腔标签+膨胀后的冠脉标签（闭运算）对心脏区域整体进行收缩、舒张

LATER 增加整体微小刚体运动效果，模拟身体移动与呼吸
LATER 增加造影剂注入和衰减效果