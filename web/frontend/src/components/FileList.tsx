import { useMemo, useState } from "react";
import { Input, List, Segmented, Slider, Tag, InputNumber } from "antd";
import {
  CheckCircleFilled,
  FileImageOutlined,
  VideoCameraOutlined,
} from "@ant-design/icons";
import { useStudio } from "../store/useStudio";

type FilterMode = "all" | "labeled" | "unlabeled" | "empty";

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
  const { images, video, currentIndex, selectImage } = useStudio();
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<FilterMode>("all");

  const matchFilter = (im: { has_label: boolean; shape_count: number | null }) => {
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

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return images
      .map((im, i) => ({ ...im, index: i }))
      .filter(matchFilter)
      .filter((im) => !q || im.filename.toLowerCase().includes(q));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [images, query, filter]);

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
