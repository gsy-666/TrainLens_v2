import { useCallback, useEffect, useRef, useState } from "react";
import { Button, Collapse, message, Progress, Tag, Tooltip } from "antd";
import { CaretRightOutlined, StopOutlined } from "@ant-design/icons";
import * as api from "../api/client";
import { useStudio } from "../store/useStudio";

// 每类问题列表最多展示的条数，超出部分折叠为「…等 N 张/组」
const MAX_ITEMS = 50;

const ISSUE_TEXT: Record<string, string> = {
  out_of_bounds: "框越界",
  oversized: "框过大",
  tiny: "框过小",
  degenerate: "形状退化",
};

function fmtScanTime(s: string): string {
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) return s;
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${hh}:${mm}`;
}

interface Props {
  /** 当前标注目录；变化时重置扫描进度与报告 */
  dir: string | null;
  onBack: () => void;
}

export default function DatasetHealth({ dir, onBack }: Props) {
  const [report, setReport] = useState<api.DatasetHealthReport | null>(null);
  const [scan, setScan] = useState<api.HealthScanStatus | null>(null);
  const [starting, setStarting] = useState(false);
  const timerRef = useRef<number | undefined>(undefined);

  const stopPolling = useCallback(() => {
    window.clearInterval(timerRef.current);
    timerRef.current = undefined;
  }, []);

  const loadReport = useCallback(async () => {
    try {
      const r = await api.getDatasetHealthReport();
      setReport(r.updated_at ? r : null);
    } catch {
      /* ignore */
    }
  }, []);

  const startPolling = useCallback(() => {
    stopPolling();
    timerRef.current = window.setInterval(async () => {
      try {
        const s = await api.getHealthScanStatus();
        setScan(s);
        if (!s.running) {
          stopPolling();
          if (s.done) {
            message.success("健康扫描完成");
            await loadReport();
          }
        }
      } catch {
        /* server unreachable */
      }
    }, 1000);
  }, [loadReport, stopPolling]);

  // 首次挂载 / 目录变化：重置后拉取已有报告（不自动开扫）；
  // 若后端正在扫描（例如页面切换前启动的），接管进度轮询
  useEffect(() => {
    setReport(null);
    setScan(null);
    void loadReport();
    api
      .getHealthScanStatus()
      .then((s) => {
        setScan(s);
        if (s.running) startPolling();
      })
      .catch(() => undefined);
    return stopPolling;
  }, [dir, loadReport, startPolling, stopPolling]);

  const onStart = useCallback(async () => {
    setStarting(true);
    try {
      await api.startHealthScan();
      setScan({ running: true, processed: 0, total: 0, done: false, error: null, error_count: 0 });
      startPolling();
    } catch (e) {
      const err = e as { response?: { status?: number; data?: { detail?: string } }; message: string };
      const status = err.response?.status;
      if (status === 400) {
        message.warning("请先在标注页打开图片目录");
      } else if (status === 409) {
        message.info("扫描已在进行中");
        startPolling();
      } else {
        message.error(`启动扫描失败: ${err.response?.data?.detail ?? err.message}`);
      }
    } finally {
      setStarting(false);
    }
  }, [startPolling]);

  const onStop = useCallback(async () => {
    try {
      await api.stopHealthScan();
      message.info("已请求停止");
    } catch {
      /* ignore */
    }
  }, []);

  const onPickImage = useCallback(
    async (name: string) => {
      const found = await useStudio.getState().selectImageByName(name);
      if (!found) {
        message.warning("当前目录中找不到该图片");
        return;
      }
      onBack();
    },
    [onBack]
  );

  const running = !!scan?.running;
  const pct = scan && scan.total > 0 ? Math.round((scan.processed / scan.total) * 100) : 0;

  // 可点击图名：跳回标注页并选中该图
  const link = (name: string) => (
    <a onClick={() => void onPickImage(name)}>{name}</a>
  );
  const joinNames = (names: string[]) =>
    names.map((n, i) => (
      <span key={n}>
        {i > 0 && "，"}
        {link(n)}
      </span>
    ));
  const moreLine = (total: number, unit: string) =>
    total > MAX_ITEMS ? (
      <div style={{ color: "#999" }}>…等 {total} {unit}</div>
    ) : null;

  const summary = report?.summary;
  const dupGroups = report?.duplicate_groups ?? [];
  const blurry = report?.blurry ?? [];
  const dark = report?.dark ?? [];
  const bright = report?.bright ?? [];
  const shapeIssues = report?.shape_issues ?? [];
  const dupCount = summary?.duplicate_groups ?? dupGroups.length;
  const blurryCount = summary?.blurry ?? blurry.length;
  const darkCount = summary?.dark ?? dark.length;
  const brightCount = summary?.bright ?? bright.length;
  const shapeCount = summary?.shape_issue_images ?? shapeIssues.length;
  const allHealthy =
    dupCount + blurryCount + darkCount + brightCount + shapeCount === 0;

  const listStyle: React.CSSProperties = {
    display: "flex",
    flexDirection: "column",
    gap: 4,
    fontSize: 12,
  };

  const collapseItems: NonNullable<React.ComponentProps<typeof Collapse>["items"]> = [];
  if (dupGroups.length > 0) {
    collapseItems.push({
      key: "dup",
      label: `重复 / 相似图（${dupCount} 组）`,
      children: (
        <div style={listStyle}>
          {dupGroups.slice(0, MAX_ITEMS).map((g, i) => (
            <div key={i}>
              <span style={{ color: "#999" }}>
                组 {i + 1}（{g.length} 张）：
              </span>
              {joinNames(g)}
            </div>
          ))}
          {moreLine(dupGroups.length, "组")}
        </div>
      ),
    });
  }
  if (blurry.length > 0) {
    collapseItems.push({
      key: "blurry",
      label: `模糊图（${blurryCount} 张，越糊越靠前）`,
      children: (
        <div style={listStyle}>
          {blurry.slice(0, MAX_ITEMS).map((b) => (
            <div key={b.name}>
              {link(b.name)}
              <span style={{ color: "#999", marginLeft: 8 }}>得分 {b.score}</span>
            </div>
          ))}
          {moreLine(blurry.length, "张")}
        </div>
      ),
    });
  }
  if (dark.length > 0) {
    collapseItems.push({
      key: "dark",
      label: `过暗图（${darkCount} 张）`,
      children: (
        <div style={listStyle}>
          {dark.slice(0, MAX_ITEMS).map((b) => (
            <div key={b.name}>
              {link(b.name)}
              <span style={{ color: "#999", marginLeft: 8 }}>亮度 {b.brightness}</span>
            </div>
          ))}
          {moreLine(dark.length, "张")}
        </div>
      ),
    });
  }
  if (bright.length > 0) {
    collapseItems.push({
      key: "bright",
      label: `过亮图（${brightCount} 张）`,
      children: (
        <div style={listStyle}>
          {bright.slice(0, MAX_ITEMS).map((b) => (
            <div key={b.name}>
              {link(b.name)}
              <span style={{ color: "#999", marginLeft: 8 }}>亮度 {b.brightness}</span>
            </div>
          ))}
          {moreLine(bright.length, "张")}
        </div>
      ),
    });
  }
  if (shapeIssues.length > 0) {
    collapseItems.push({
      key: "shape",
      label: `标注异常（${shapeCount} 张）`,
      children: (
        <div style={listStyle}>
          {shapeIssues.slice(0, MAX_ITEMS).map((it) => (
            <div key={it.name}>
              {link(it.name)}
              <div style={{ marginLeft: 8, color: "#666" }}>
                {it.issues.map((iss, j) => (
                  <div key={j}>
                    {iss.label}：{ISSUE_TEXT[iss.issue] ?? iss.issue}
                    {iss.detail ? `（${iss.detail}）` : ""}
                  </div>
                ))}
              </div>
            </div>
          ))}
          {moreLine(shapeIssues.length, "张")}
        </div>
      ),
    });
  }

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ fontWeight: 600 }}>健康扫描</span>
        {report?.updated_at && (
          <span style={{ fontSize: 12, color: "#999" }}>
            扫描于 {fmtScanTime(report.updated_at)}
          </span>
        )}
      </div>
      <div style={{ marginTop: 8 }}>
        {!running ? (
          <Tooltip title="检查重复图、模糊图、过暗过亮图和标注异常">
            <Button
              size="small"
              type="primary"
              ghost
              icon={<CaretRightOutlined />}
              loading={starting}
              onClick={onStart}
            >
              开始健康扫描
            </Button>
          </Tooltip>
        ) : (
          <Button size="small" danger icon={<StopOutlined />} onClick={onStop}>
            停止
          </Button>
        )}
      </div>
      <div style={{ fontSize: 12, color: "#999", marginTop: 6 }}>
        扫描当前标注目录，点击报告中的图名可跳回标注页查看。
      </div>
      {running && scan && (
        <div style={{ marginTop: 8 }}>
          <Progress percent={pct} size="small" />
          <div style={{ fontSize: 12, color: "#999" }}>
            已处理 {scan.processed} / {scan.total || "?"} 张
          </div>
        </div>
      )}
      {!running && scan?.error && (
        <div style={{ fontSize: 12, color: "#f5222d", marginTop: 6 }}>
          扫描出错：{scan.error}
        </div>
      )}
      {report && (
        <>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 10 }}>
            {allHealthy ? (
              <Tag color="green">数据健康</Tag>
            ) : (
              <>
                {dupCount > 0 && <Tag color="orange">重复 {dupCount} 组</Tag>}
                {blurryCount > 0 && <Tag color="orange">模糊 {blurryCount} 张</Tag>}
                {darkCount > 0 && <Tag color="orange">过暗 {darkCount} 张</Tag>}
                {brightCount > 0 && <Tag color="orange">过亮 {brightCount} 张</Tag>}
                {shapeCount > 0 && <Tag color="red">标注异常 {shapeCount} 张</Tag>}
              </>
            )}
          </div>
          {collapseItems.length > 0 && (
            <Collapse size="small" style={{ marginTop: 8 }} items={collapseItems} />
          )}
        </>
      )}
    </div>
  );
}
