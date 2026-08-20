import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Button,
  Input,
  InputNumber,
  List,
  message,
  Modal,
  Progress,
  Segmented,
  Select,
  Slider,
  Tag,
  Tooltip,
} from "antd";
import {
  CheckCircleFilled,
  ExperimentOutlined,
  FileImageOutlined,
  VideoCameraOutlined,
} from "@ant-design/icons";
import * as api from "../api/client";
import { useStudio } from "../store/useStudio";

type FilterMode = "all" | "labeled" | "unlabeled" | "empty";
type SortMode = "default" | "hard";

/** 难例分数段颜色：0-0.3 红、0.3-0.7 橙、>0.7 绿 */
function scoreColor(score: number): string {
  if (score <= 0.3) return "red";
  if (score <= 0.7) return "orange";
  return "green";
}

const FILTER_OPTIONS = [
  { value: "all", label: "全部" },
  { value: "labeled", label: "已标注" },
  { value: "unlabeled", label: "未标注" },
  { value: "empty", label: "空标注" },
];

function VideoPanel() {
  const { video, currentIndex, selectImage } = useStudio();
  if (!video) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ padding: 8, borderBottom: "1px solid #f0f0f0" }}>
        <div
          style={{
            fontSize: 12,
            color: "#666",
            marginBottom: 8,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
          title={video.path}
        >
          <VideoCameraOutlined /> {video.path.split(/[/\\]/).pop()}
        </div>
        <Slider
          min={0}
          max={Math.max(0, video.frameCount - 1)}
          value={currentIndex}
          onChange={(v) => selectImage(v)}
          tooltip={{ formatter: (v) => `第 ${(v ?? 0) + 1} 帧` }}
        />
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 12, color: "#888" }}>跳转</span>
          <InputNumber
            size="small"
            min={1}
            max={video.frameCount}
            value={currentIndex + 1}
            onChange={(v) => v && selectImage(v - 1)}
            style={{ width: 80 }}
          />
          <span style={{ fontSize: 12, color: "#888" }}>
            / {video.frameCount} 帧{video.fps ? ` · ${video.fps.toFixed(1)}fps` : ""}
          </span>
        </div>
      </div>
      <div style={{ flex: 1, overflow: "auto" }}>
        <div style={{ padding: "8px 12px", fontSize: 12, color: "#999" }}>
          已标注帧（{video.labeledFrames.length}）
        </div>
        <List
          size="small"
          dataSource={video.labeledFrames}
          renderItem={(f) => (
            <List.Item
              onClick={() => selectImage(f)}
              style={{
                cursor: "pointer",
                padding: "4px 16px",
                background: f === currentIndex ? "#e6f4ff" : undefined,
              }}
            >
              <CheckCircleFilled style={{ color: "#52c41a", marginRight: 8 }} />
              第 {f + 1} 帧
            </List.Item>
          )}
        />
      </div>
    </div>
  );
}

export default function FileList() {
  const { dir, images, video, currentIndex, selectImage } = useStudio();
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<FilterMode>("all");
  const [selectedLabel, setSelectedLabel] = useState<string>("all");

  // ---- 主动学习：难例扫描 ------------------------------------------------
  const [sortMode, setSortMode] = useState<SortMode>("default");
  const [scores, setScores] = useState<Record<string, api.ALScoreEntry>>({});
  const [scan, setScan] = useState<api.ALScanStatus | null>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const scanTimerRef = useRef<number | undefined>(undefined);

  const scanning = !!scan?.running;

  const startScanPolling = useCallback(() => {
    window.clearInterval(scanTimerRef.current);
    scanTimerRef.current = window.setInterval(async () => {
      try {
        const s = await api.getALScanStatus();
        setScan(s);
        if (!s.running) {
          window.clearInterval(scanTimerRef.current);
          scanTimerRef.current = undefined;
          const sc = await api.getALScores();
          setScores(sc.scores);
          setSortMode("hard");
          if (s.error) {
            message.error(`难例扫描出错: ${s.error}`);
          } else if (s.error_count > 0) {
            message.warning(`扫描完成，${s.error_count} 张失败，已切换为难例优先`);
          } else {
            message.success("扫描完成，已切换为难例优先排序");
          }
        }
      } catch {
        /* server unreachable */
      }
    }, 1000);
  }, []);

  // 挂载时恢复服务端已有的扫描状态/分数（如页面切换后回来）；
  // 已结束的扫描不恢复进度条，避免展示其它目录的残留状态
  useEffect(() => {
    api
      .getALScores()
      .then((d) => setScores(d.scores))
      .catch(() => undefined);
    api
      .getALScanStatus()
      .then((s) => {
        if (s.running) {
          setScan(s);
          startScanPolling();
        }
      })
      .catch(() => undefined);
    return () => window.clearInterval(scanTimerRef.current);
  }, [startScanPolling]);

  // 轮询模型加载状态，控制扫描按钮可用性
  useEffect(() => {
    let stopped = false;
    const check = async () => {
      try {
        const s = await api.getModelStatus();
        if (!stopped) setModelLoaded(!!s.loaded);
      } catch {
        /* ignore */
      }
    };
    check();
    const t = window.setInterval(check, 5000);
    return () => {
      stopped = true;
      window.clearInterval(t);
    };
  }, []);

  // 打开新目录时清空本地难度分状态（含旧目录遗留的扫描轮询）
  useEffect(() => {
    window.clearInterval(scanTimerRef.current);
    scanTimerRef.current = undefined;
    setScores({});
    setScan(null);
    setSortMode("default");
  }, [dir]);

  const beginScan = useCallback(
    async (scope: "unlabeled" | "all") => {
      try {
        const r = await api.startALScan(undefined, scope);
        setScan({ running: true, processed: 0, total: r.total, done: false, error: null, error_count: 0 });
        startScanPolling();
      } catch (e) {
        const err = e as { response?: { data?: { detail?: string } }; message: string };
        message.error(`启动扫描失败: ${err.response?.data?.detail ?? err.message}`);
      }
    },
    [startScanPolling]
  );

  const onScanClick = useCallback(() => {
    const unlabeled = images.filter((im) => !im.has_label).length;
    if (unlabeled === 0) {
      Modal.confirm({
        title: "全部图片均已标注",
        content: "没有未标注图片可扫描，是否改为扫描全部图片？",
        okText: "扫描全部",
        cancelText: "取消",
        onOk: () => beginScan("all"),
      });
    } else {
      beginScan("unlabeled");
    }
  }, [images, beginScan]);

  const onStopScan = useCallback(async () => {
    try {
      await api.stopALScan();
      message.info("已请求停止扫描");
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      message.error(`停止失败: ${err.response?.data?.detail ?? err.message}`);
    }
  }, []);

  const onClearScores = useCallback(async () => {
    try {
      await api.clearALScores();
      setScores({});
      setSortMode("default");
      message.success("已清除难度分");
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      message.error(`清除失败: ${err.response?.data?.detail ?? err.message}`);
    }
  }, []);

  const matchFilter = (im: { has_label: boolean; shape_count: number | null; labels?: string[] }) => {
    switch (filter) {
      case "labeled":
        return im.has_label && (im.shape_count ?? 0) > 0;
      case "unlabeled":
        return !im.has_label;
      case "empty":
        return im.has_label && (im.shape_count ?? 0) === 0;
      default:
        return true;
    }
  };

  const counts = useMemo(() => {
    let labeled = 0, unlabeled = 0, empty = 0;
    for (const im of images) {
      if (!im.has_label) unlabeled++;
      else if ((im.shape_count ?? 0) === 0) empty++;
      else labeled++;
    }
    return { labeled, unlabeled, empty };
  }, [images]);

  const labelOptions = useMemo(() => {
    const labels = new Set<string>();
    for (const im of images) {
      for (const label of im.labels ?? []) labels.add(label);
    }
    return ["all", ...Array.from(labels).sort()];
  }, [images]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = images
      .map((im, i) => ({ ...im, index: i }))
      .filter(matchFilter)
      .filter((im) => (selectedLabel === "all" ? true : (im.labels ?? []).includes(selectedLabel)))
      .filter((im) => !q || im.filename.toLowerCase().includes(q));
    if (sortMode === "hard") {
      // 难例优先：分数升序（越低越没把握）→ 扫描失败（null）→ 无扫描数据
      const rank = (fn: string) => {
        const s = scores[fn];
        if (!s) return 2;
        if (s.score == null) return 1;
        return 0;
      };
      list.sort((a, b) => {
        const ra = rank(a.filename);
        const rb = rank(b.filename);
        if (ra !== rb) return ra - rb;
        if (ra === 0) {
          const sa = scores[a.filename].score as number;
          const sb = scores[b.filename].score as number;
          if (sa !== sb) return sa - sb;
        }
        return a.index - b.index;
      });
    }
    return list;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [images, query, filter, selectedLabel, sortMode, scores]);

  if (video) return <VideoPanel />;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ padding: 8, borderBottom: "1px solid #f0f0f0" }}>
        <Input.Search
          size="small"
          placeholder="搜索图片"
          allowClear
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ marginBottom: 6 }}
        />
        <Select
          size="small"
          style={{ width: "100%", marginBottom: 6 }}
          value={selectedLabel}
          onChange={setSelectedLabel}
          options={labelOptions.map((label) => ({ value: label, label: label === "all" ? "全部类别" : label }))}
        />
        <Segmented
          size="small"
          block
          value={filter}
          onChange={(v) => setFilter(v as FilterMode)}
          options={FILTER_OPTIONS.map((o) => ({
            ...o,
            label:
              o.value === "all"
                ? `全部 ${images.length}`
                : o.value === "labeled"
                  ? `已标注 ${counts.labeled}`
                  : o.value === "unlabeled"
                    ? `未标注 ${counts.unlabeled}`
                    : `空标注 ${counts.empty}`,
          }))}
        />
        <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
          <Select
            size="small"
            style={{ flex: 1 }}
            value={sortMode}
            onChange={(v) => setSortMode(v as SortMode)}
            options={[
              { value: "default", label: "默认排序" },
              { value: "hard", label: "难例优先" },
            ]}
          />
          <Tooltip
            title={
              modelLoaded
                ? "用已加载的 AI 模型评估每张图的标注难度，最没把握的排前面"
                : "请先在右侧加载 AI 模型"
            }
          >
            <span>
              {scanning ? (
                <Button size="small" danger onClick={onStopScan}>
                  停止扫描
                </Button>
              ) : (
                <Button
                  size="small"
                  icon={<ExperimentOutlined />}
                  disabled={!modelLoaded || images.length === 0}
                  onClick={onScanClick}
                >
                  难例扫描
                </Button>
              )}
            </span>
          </Tooltip>
        </div>
        {scan && (scan.running || scan.done) && (
          <div style={{ marginTop: 6 }}>
            <Progress
              size="small"
              percent={scan.total > 0 ? Math.round((scan.processed / scan.total) * 100) : 0}
              status={scan.error ? "exception" : scan.running ? "active" : undefined}
            />
            <div style={{ fontSize: 11, color: "#999" }}>
              {scan.running ? "扫描中" : "扫描结束"} {scan.processed}/{scan.total}
              {scan.error_count > 0 ? ` · ${scan.error_count} 张失败` : ""}
            </div>
          </div>
        )}
        {Object.keys(scores).length > 0 && !scanning && (
          <Button
            size="small"
            type="link"
            style={{ padding: 0, marginTop: 4, fontSize: 12 }}
            onClick={onClearScores}
          >
            清除难度分
          </Button>
        )}
      </div>
      <div style={{ flex: 1, overflow: "auto" }}>
        <List
          size="small"
          dataSource={filtered}
          renderItem={(item) => (
            <List.Item
              onClick={() => selectImage(item.index)}
              style={{
                cursor: "pointer",
                padding: "6px 12px",
                background: item.index === currentIndex ? "#e6f4ff" : undefined,
                borderLeft:
                  item.index === currentIndex
                    ? "3px solid #1677ff"
                    : "3px solid transparent",
              }}
            >
              <FileImageOutlined style={{ marginRight: 8, color: "#999" }} />
              <span
                style={{
                  flex: 1,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
                title={item.filename}
              >
                {item.filename}
              </span>
              {sortMode === "hard" &&
                scores[item.filename] &&
                (() => {
                  const s = scores[item.filename];
                  const tagStyle = { marginLeft: 4, fontSize: 10, lineHeight: "14px", padding: "0 4px" };
                  if (s.score == null)
                    return (
                      <Tag color="red" style={tagStyle}>
                        失败
                      </Tag>
                    );
                  if (s.count === 0)
                    return <Tag style={tagStyle}>无检测</Tag>;
                  return (
                    <Tag color={scoreColor(s.score)} style={tagStyle}>
                      {s.score.toFixed(2)}
                    </Tag>
                  );
                })()}
              {item.has_label && (item.shape_count ?? 0) > 0 && (
                <span style={{ color: "#52c41a", fontSize: 11, marginLeft: 4 }}>
                  {item.shape_count}
                </span>
              )}
              {item.has_label && (item.shape_count ?? 0) === 0 && (
                <Tag color="orange" style={{ marginLeft: 4, fontSize: 10, lineHeight: "14px", padding: "0 4px" }}>
                  空
                </Tag>
              )}
              {item.has_label && <CheckCircleFilled style={{ color: "#52c41a", marginLeft: 4 }} />}
            </List.Item>
          )}
        />
      </div>
      <div style={{ padding: 8, borderTop: "1px solid #f0f0f0", color: "#999", fontSize: 12 }}>
        共 {filtered.length} 张
      </div>
    </div>
  );
}
