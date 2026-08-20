/** Deterministic color per label, similar in spirit to the desktop colormap. */

export function labelColor(label: string): string {
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = (hash * 31 + label.charCodeAt(i)) >>> 0;
  }
  const hue = hash % 360;
  return `hsl(${hue}, 85%, 50%)`;
}

export function withAlpha(color: string, alpha: number): string {
  const m = color.match(/hsl\((\d+),\s*(\d+)%,\s*(\d+)%\)/);
  if (m) {
    return `hsla(${m[1]}, ${m[2]}%, ${m[3]}%, ${alpha})`;
  }
  return color;
}
