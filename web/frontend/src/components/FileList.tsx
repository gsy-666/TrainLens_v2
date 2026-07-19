import { useMemo, useState } from "react";
import { Input, InputNumber, List, Slider } from "antd";
import {
  CheckCircleFilled,
  FileImageOutlined,
  VideoCameraOutlined,
} from "@ant-design/icons";
import { useStudio } from "../store/useStudio";

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

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return images
      .map((im, i) => ({ ...im, index: i }))
      .filter((im) => !q || im.filename.toLowerCase().includes(q));
  }, [images, query]);

  if (video) return <VideoPanel />;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ padding: 8 }}>
        <Input.Search
          size="small"
          placeholder="搜索图片"
          allowClear
          value={query}
          onChange={(e) => setQuery(e.target.value)}
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
              {item.has_label && (
                <CheckCircleFilled style={{ color: "#52c41a", marginLeft: 4 }} />
              )}
            </List.Item>
          )}
        />
      </div>
      <div style={{ padding: 8, borderTop: "1px solid #f0f0f0", color: "#999", fontSize: 12 }}>
        共 {images.length} 张
      </div>
    </div>
  );
}
