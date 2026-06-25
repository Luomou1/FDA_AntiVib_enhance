source visual truth path: user-provided images #1-#5, `E:/xwechat_files/wxid_rt72htp376r222_f097/temp/RWTemp/2026-06/9e20f478899dc29eb19741386f9343c8/`
implementation screenshot path: `.superpowers/brainstorm/ui-redesign-20260609/product-design-hard-neutral-console-native.png`
viewport: 1400 x 900
state: initial no-data workbench, hard neutral flat-minimal console theme
full-view comparison evidence: Reference shows a measurement console with clear application chrome, large central analysis viewport, right-side parameter/control panel, workflow controls, and compact bottom status. After user feedback, implementation intentionally simplifies the visual logic and hardens the visual language: one top-level navigation row, one workflow step rail, one right-side parameter/action panel, one central viewport, one bottom status bar, graphite top chrome, neutral white/gray surfaces, square controls, smaller type, and a single restrained blue accent. Duplicate top tool-strip actions, duplicate run controls, purple-gray surfaces, soft rounded blocks, and mixed saturated accents were removed.
focused region comparison evidence: Focused region crops were not needed for this pass because the native screenshot is readable at full size and the main fidelity risk is macro layout/chrome language rather than icon-perfect cloning. The offscreen screenshot path `.superpowers/brainstorm/ui-redesign-20260609/product-design-industrial-console.png` was not used for text judgment because Qt offscreen reported zero available font families and rendered CJK as boxes.

**Findings**
- No actionable P0/P1/P2 findings remain.

**Required Fidelity Surfaces**
- Fonts and typography: Native Qt screenshot renders Chinese labels correctly with the Windows font stack. The base UI size is reduced to 12px, with compact headers and dense form labels to avoid the previous soft, web-form feel.
- Spacing and layout rhythm: Layout now uses a flatter hierarchy: compact top navigation, left workflow rail, central viewport, right parameter column, bottom status. Repeated navigation/action rows were removed, and controls use tighter vertical rhythm.
- Colors and visual tokens: Theme now uses a hard neutral instrument palette: graphite top chrome, white/off-white work surfaces, gray borders, slate text, and one restrained blue for active state/action. The previous gray-purple background, pale soft blocks, and cyan/blue mix were removed.
- Image quality and asset fidelity: The source UI is mostly application chrome and data canvases rather than bitmap artwork. No placeholder imagery was introduced. Existing Qt icons remain platform-independent for the left rail.
- Copy and content: App-specific scientific workflow labels are preserved: data source, scanning, algorithm configuration, K0, 2D/3D views, logs, and export.

**Patches Made**
- Replaced the previous menu/tool-strip combination with a single flat top navigation bar.
- Removed duplicate top run actions; the right parameter panel is now the only run/action location.
- Removed the left global icon rail from the visible layout; top navigation is the only primary page switcher.
- Switched default theme mode to light and rebuilt QSS tokens around neutral surfaces.
- Reduced font sizes, button heights, radius, and selected-state softness.
- Adjusted right parameter panel width, spacing, border radius, and central canvas text contrast.
- Updated GUI tests to cover the flat top navigation, default light mode, and non-duplicated run controls.

**Implementation Checklist**
- Hard neutral flat theme applied.
- Flat top primary navigation available.
- Duplicate tool strip removed.
- Central result viewport remains dominant.
- Right parameter panel remains functional.
- Existing controls, callbacks, and result canvases preserved.
- GUI and full test suite pass.

**Follow-up Polish**
- Replace the current left-rail generated line icons with a real bundled icon set if icon-level fidelity becomes a priority.
- Add a richer no-data preview texture or sample surface only if the application gains bundled demo data.

final result: passed
