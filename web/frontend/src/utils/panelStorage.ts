/**
 * 侧边栏宽度的 localStorage 记忆。
 * 用法：Splitter.Panel 的 defaultSize 读 readPanelWidth（仅挂载时生效），
 * Splitter 的 onResize 里 savePanelWidth 持久化；面板不做受控，避免拖拽卡顿。
 */

export function readPanelWidth(key: string, fallback: number): number {
  try {
    const v = Number(localStorage.getItem(key));
    return Number.isFinite(v) && v > 0 ? v : fallback;
  } catch {
    return fallback;
  }
}

export function savePanelWidth(key: string, width: number): void {
  try {
    if (Number.isFinite(width) && width > 0) {
      localStorage.setItem(key, String(Math.round(width)));
    }
  } catch {
    /* localStorage 不可用时静默忽略 */
  }
}
