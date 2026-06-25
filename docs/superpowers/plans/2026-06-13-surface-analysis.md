# 平面与台阶分析模块实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有桌面应用中增加相互独立的平面分析和台阶分析一级页面，并迁移三个 MATLAB 程序的核心算法与绘图。

**Architecture:** 使用纯算法模块承载矩阵处理和统计，使用独立 Qt 页面承载文件选择、三点/矩形交互和结果图。主窗口只增加页面注册、导航和资源释放，避免继续堆积业务逻辑。

**Tech Stack:** Python 3.11、NumPy、SciPy、Matplotlib、PySide6、pytest、pytest-qt

---

### Task 1: 建立核心数据模型和矩阵读取

**Files:**
- Create: `app/core/surface_analysis.py`
- Create: `tests/test_surface_analysis.py`

- [ ] 写入 `load_height_matrix()` 的失败测试，覆盖二维矩阵、空文件和非二维输入。
- [ ] 运行 `pytest tests/test_surface_analysis.py -v`，确认缺少模块导致失败。
- [ ] 实现文本矩阵读取、有限值校验和局部中值 NaN 修复。
- [ ] 再次运行定向测试，确认读取与修复通过。

### Task 2: 实现平面分析算法

**Files:**
- Modify: `app/core/surface_analysis.py`
- Modify: `tests/test_surface_analysis.py`

- [ ] 用带已知 X/Y 倾斜和局部起伏的合成矩阵编写失败测试。
- [ ] 实现简单平面、MAD 鲁棒平面和二次曲面拟合。
- [ ] 实现极端异常点修复、五倍标准差残余去噪和均值回填。
- [ ] 断言校正斜率、输出形状、高度范围和标准差。

### Task 3: 实现台阶分析算法

**Files:**
- Modify: `app/core/surface_analysis.py`
- Modify: `tests/test_surface_analysis.py`

- [ ] 用已知倾斜、已知台阶高度和离群点的合成矩阵编写失败测试。
- [ ] 实现三点平面构造，并拒绝共线或退化参考点。
- [ ] 实现平滑直方图双峰阈值、分位数回退和形态学清理。
- [ ] 实现按层三倍标准差去噪及“不去噪”旁路。
- [ ] 实现矩形裁剪、区域均值和台阶高度计算。
- [ ] 运行定向测试，确认两种模式保持同一分层且去噪结果存在差异。

### Task 4: 实现通用分析画布和页面

**Files:**
- Create: `app/gui/surface_analysis_page.py`
- Modify: `app/gui/theme.py`
- Create: `tests/test_surface_analysis_page.py`

- [ ] 编写页面结构测试，断言平面页和台阶页各自拥有文件输入且状态不共享。
- [ ] 实现平面页参数卡、指标卡、结果页签和行/列轮廓控制。
- [ ] 实现台阶页模式选择、三点点击画布、双矩形选择和结果页签。
- [ ] 为交互控件增加可访问名称、禁用态、明确提示和错误恢复文本。
- [ ] 补充 QSS，使新页面复用现有主题令牌并保持 4/8 像素间距节奏。

### Task 5: 接入一级导航

**Files:**
- Modify: `app/gui/main_window.py`
- Modify: `tests/test_gui_layout.py`

- [ ] 将 `main_pages` 顺序调整为工作台、平面分析、台阶分析、设置。
- [ ] 将顶部按钮顺序调整为 `工作台 | 平面分析 | 台阶分析 | 设置`。
- [ ] 更新设置页索引相关测试，验证四个按钮与四个页面一一对应。
- [ ] 在主窗口关闭时释放两个新页面中的 Matplotlib 选择器和画布引用。

### Task 6: 完成验证

**Files:**
- Modify: `README.md`

- [ ] 更新功能概览和运行说明，注明两个模块读取独立高度文本文件。
- [ ] 运行 `pytest tests/test_surface_analysis.py tests/test_surface_analysis_page.py tests/test_gui_layout.py -v`。
- [ ] 运行 `pytest -q`。
- [ ] 运行 `python -m compileall app tests`。
- [ ] 检查 `git diff --check`，确认没有空白错误。
- [ ] 按设计清单逐项核对导航、模式、绘图、交互和错误反馈。

## 执行结果

- 状态：已完成实现与验证。
- 顶部导航：`工作台 | 平面分析 | 台阶分析 | 设置`。
- 定向覆盖：核心算法、页面独立状态、去噪/不去噪、三点调平、双区域测量和导航切换。
- 最终证据：以本次执行结束前最后一次全量测试、编译检查和 `git diff --check` 输出为准。
