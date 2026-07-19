/** Minimal dependency-free SVG line chart for training metrics. */

interface Series {
  name: string;
  points: [number, number][];
}

interface Props {
  series: Series[];
  width?: number;
  height?: number;
  title?: string;
}

const COLORS = [
  "#1677ff",
  "#52c41a",
  "#fa8c16",
  "#eb2f96",
  "#722ed1",
  "#13c2c2",
  "#f5222d",
  "#a0d911",
];

export default function LineChart({ series, width = 560, height = 220, title }: Props) {
  const pad = { l: 48, r: 12, t: title ? 28 : 12, b: 24 };
  const w = width - pad.l - pad.r;
  const h = height - pad.t - pad.b;

  const all = series.flatMap((s) => s.points);
  if (all.length === 0) {
    return (
      <div style={{ width, height, display: "flex", alignItems: "center", justifyContent: "center", color: "#bbb", border: "1px dashed #e0e0e0", borderRadius: 6 }}>
        暂无数据
      </div>
    );
  }
  const xMin = Math.min(...all.map((p) => p[0]));
  const xMax = Math.max(...all.map((p) => p[0]));
  const yMin = Math.min(...all.map((p) => p[1]));
  const yMax = Math.max(...all.map((p) => p[1]));
  const xSpan = xMax - xMin || 1;
  const ySpan = yMax - yMin || 1e-9;

  const sx = (x: number) => pad.l + ((x - xMin) / xSpan) * w;
  const sy = (y: number) => pad.t + h - ((y - yMin) / ySpan) * h;

  const yTicks = 4;
  const ticks = Array.from({ length: yTicks + 1 }, (_, i) => yMin + (ySpan * i) / yTicks);

  return (
    <div>
      {title && <div style={{ fontWeight: 600, marginBottom: 4 }}>{title}</div>}
      <svg width={width} height={height} style={{ background: "#fafafa", borderRadius: 6 }}>
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={pad.l} y1={sy(t)} x2={pad.l + w} y2={sy(t)} stroke="#eee" />
            <text x={pad.l - 6} y={sy(t) + 4} fontSize={10} fill="#999" textAnchor="end">
              {Math.abs(t) < 0.001 ? t.toExponential(0) : t.toFixed(3)}
            </text>
          </g>
        ))}
        <text x={pad.l + w / 2} y={height - 6} fontSize={10} fill="#999" textAnchor="middle">
          epoch
        </text>
        {series.map((s, si) => (
          <polyline
            key={s.name}
            fill="none"
            stroke={COLORS[si % COLORS.length]}
            strokeWidth={1.5}
            points={s.points.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ")}
          />
        ))}
      </svg>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: 4 }}>
        {series.map((s, si) => (
          <span key={s.name} style={{ fontSize: 11, color: "#666" }}>
            <span
              style={{
                display: "inline-block",
                width: 10,
                height: 3,
                background: COLORS[si % COLORS.length],
                marginRight: 4,
                verticalAlign: "middle",
              }}
            />
            {s.name}
          </span>
        ))}
      </div>
    </div>
  );
}
