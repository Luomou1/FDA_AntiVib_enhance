# Guided Analysis Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 FDA Antivib 主窗口重构为浅色优先、支持系统主题、采用双层导航和四步参数流程的 Guided Analysis Console。

**Architecture:** 保留 `MainWindow` 现有业务字段、线程调度、结果刷新和导出回调，仅替换界面组合层。主题集中到 `app/gui/theme.py`，轻量导航控件集中到 `app/gui/widgets.py`，主窗口使用 `QStackedWidget` 组织一级页面与参数步骤，避免触碰核心算法和共享数据结构。

**Tech Stack:** Python 3.11、PySide6、Matplotlib、PyVista、pytest、pytest-qt。

---

### Task 1: 建立主题系统

**Files:**
- Create: `app/gui/theme.py`
- Test: `tests/test_gui_layout.py`

- [ ] **Step 1: 编写主题测试**

新增测试，验证主题模块至少提供 `system`、`light`、`dark` 三种模式，并且生成的 QSS 包含主窗口、导航、参数面板、按钮、输入框、标签页、状态栏和滚动条选择器。

```python
@pytest.mark.parametrize("mode", ["light", "dark"])
def test_build_stylesheet_contains_console_selectors(mode: str) -> None:
    stylesheet = build_stylesheet(mode)
    assert "QFrame#GlobalRail" in stylesheet
    assert "QFrame#StepRail" in stylesheet
    assert "QFrame#ParameterPanel" in stylesheet
    assert "QPushButton#PrimaryButton" in stylesheet
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_gui_layout.py -q`

Expected: 因 `app.gui.theme` 尚不存在而失败。

- [ ] **Step 3: 实现主题令牌和 QSS**

`theme.py` 提供：

```python
ThemeMode = Literal["system", "light", "dark"]

def resolve_theme_mode(mode: ThemeMode, color_scheme: Qt.ColorScheme | None = None) -> Literal["light", "dark"]:
    ...

def build_stylesheet(mode: Literal["light", "dark"]) -> str:
    ...
```

主题使用集中颜色令牌，不在主窗口构建方法中散落控件级样式。系统模式通过 `QGuiApplication.styleHints().colorScheme()` 解析，识别失败时回退浅色。

- [ ] **Step 4: 运行主题测试**

Run: `python -m pytest tests/test_gui_layout.py -q`

Expected: 主题测试通过。

### Task 2: 建立导航和状态复用控件

**Files:**
- Create: `app/gui/widgets.py`
- Modify: `tests/test_gui_layout.py`

- [ ] **Step 1: 编写导航控件测试**

测试 `NavButton` 和 `StepButton` 的可选中状态、对象属性及步骤状态：

```python
def test_step_button_exposes_state(qtbot) -> None:
    button = StepButton("01", "数据来源")
    qtbot.addWidget(button)
    button.set_step_state("complete")
    assert button.property("stepState") == "complete"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_gui_layout.py -q`

Expected: 因控件尚不存在而失败。

- [ ] **Step 3: 实现复用控件**

`widgets.py` 提供：

- `NavButton`：图标文本、可选中、统一 `navRole` 属性。
- `StepButton`：序号、标题、状态标记和 `stepState` 动态属性。
- `StatusPill`：统一展示 idle/running/success/error 状态。
- `SectionHeader`：参数页标题和说明。

控件只负责呈现和信号，不持有分析业务状态。

- [ ] **Step 4: 运行控件测试**

Run: `python -m pytest tests/test_gui_layout.py -q`

Expected: 导航控件测试通过。

### Task 3: 重构主窗口布局

**Files:**
- Modify: `app/gui/main_window.py`
- Modify: `main.py`
- Test: `tests/test_gui_layout.py`

- [ ] **Step 1: 编写主窗口结构测试**

测试主窗口构建后存在：

```python
def test_main_window_uses_guided_console_layout(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.minimumWidth() == 1180
    assert window.minimumHeight() == 760
    assert window.main_pages.count() == 4
    assert window.parameter_stack.count() == 4
    assert len(window.step_buttons) == 4
```

补充步骤切换、一级页面切换、控件值保持、运行状态锁定测试。

- [ ] **Step 2: 运行结构测试确认失败**

Run: `python -m pytest tests/test_gui_layout.py -q`

Expected: 因新布局字段尚不存在而失败。

- [ ] **Step 3: 替换主窗口壳层**

在 `MainWindow._build_ui()` 中建立：

```text
TopCommandBar
└─ Body
   ├─ GlobalRail
   └─ MainPages
      ├─ WorkbenchPage
      │  ├─ StepRail
      │  ├─ ResultWorkspace
      │  └─ ParameterPanel
      ├─ ResultsPage
      ├─ LogPage
      └─ SettingsPage
BottomStatusBar
```

使用 `QButtonGroup` 管理一级导航和步骤导航，使用 `QStackedWidget` 管理一级页面和四个参数页。

- [ ] **Step 4: 将现有参数控件映射到四个步骤**

保留现有字段名与信号：

- 数据来源页：`data_source`、`folder_edit`、`folder_button`、`mat_edit`、`mat_button`。
- 扫描采样页：`start_height`、`sampling_mode`、`step_size`、`scan_log_edit`、`scan_log_button`、`scan_log_summary_label`。
- 算法配置页：K0、窗函数、补零、范围扩展、拟合、解包裹和工作流字段。
- 运行导出页：Active range、自动 K0 摘要、分析与导出按钮。

删除不再调用的旧 `_build_left_panel()` 和 `_build_left_panel_redesign()`，避免同一组控件被重复创建。

- [ ] **Step 5: 重组结果和日志区域**

保留所有画布字段及其回调。结果标签页按核心结果、三维、频谱、PhaseGap 诊断顺序组织。

完整日志 `self.log` 放入日志一级页面。底部状态栏增加 `self.latest_log_label`，`_append_log()` 同时更新完整日志和最新日志。

- [ ] **Step 6: 接入主题切换**

设置页提供 `self.theme_combo`，选项为跟随系统、浅色、深色。`MainWindow._apply_theme()` 调用 `resolve_theme_mode()` 和 `build_stylesheet()`；系统模式连接 `colorSchemeChanged` 信号。

- [ ] **Step 7: 保持运行状态和错误定位**

`_set_running_state()` 禁用参数栈和步骤导航，但不禁用结果页与日志页。参数构建失败时通过错误文本映射到对应步骤并切换页面。

- [ ] **Step 8: 运行主窗口测试**

Run: `$env:QT_QPA_PLATFORM='offscreen'; python -m pytest tests/test_gui_layout.py -q`

Expected: 主窗口、导航、步骤、主题和运行状态测试通过。

### Task 4: 回归验证与界面检查

**Files:**
- Modify when required: `app/gui/main_window.py`
- Modify when required: `app/gui/theme.py`
- Modify when required: `app/gui/widgets.py`
- Modify when required: `tests/test_gui_layout.py`

- [ ] **Step 1: 运行 GUI 定向测试**

Run: `$env:QT_QPA_PLATFORM='offscreen'; python -m pytest tests/test_gui_layout.py -q`

Expected: 全部通过，无 Qt 构建异常。

- [ ] **Step 2: 运行现有测试**

Run: `python -m pytest -q`

Expected: 全部通过，核心算法测试无回归。

- [ ] **Step 3: 运行语法编译检查**

Run: `python -m compileall app main.py`

Expected: 退出码为 0。

- [ ] **Step 4: 执行无事件循环 GUI 冒烟检查**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
python -c "from PySide6.QtWidgets import QApplication; from app.gui.main_window import MainWindow; app=QApplication.instance() or QApplication([]); w=MainWindow(); print(w.main_pages.count(), w.parameter_stack.count(), w.minimumSize().width(), w.minimumSize().height()); w.close()"
```

Expected: 输出 `4 4 1180 760`。

- [ ] **Step 5: 审查变更边界**

Run: `git diff -- app/gui main.py tests/test_gui_layout.py docs/superpowers`

确认没有修改 `app/core`、`app/pipeline` 或共享结果模型。

- [ ] **Step 6: 代码审查**

按 `requesting-code-review` 检查：

- 是否覆盖完整设计规格。
- 是否破坏现有字段和回调。
- 是否存在主题刷新、Qt 生命周期或布局尺寸问题。
- 是否缺少关键 GUI 回归测试。

修复 Critical 和 Important 问题后重新执行步骤 1 至步骤 4。

> Git 提交不属于本计划的自动操作；仓库规则要求在用户明确授权后再创建提交。
