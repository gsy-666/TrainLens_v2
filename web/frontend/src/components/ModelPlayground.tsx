import { useCallback, useEffect, useRef, useState } from "react";
import { Button, List, message, Select, Slider, Upload } from "antd";
import { CheckCircleFilled, InboxOutlined } from "@ant-design/icons";
import * as api from "../api/client";
import type { Shape } from "../types";

/** 把任意 shape 归一成可绘制的点列（矩形/旋转框/多边形直接用 points，其余取包围盒四角）。 */
function shapeOutline(s: Shape): [number, number][] {
  if (s.points.length === 0) return [];
  if (
    (s.shape_type === "rectangle" || s.shape_type === "rotation" || s.shape_type === "polygon") &&
    s.points.length >= 2
  ) {
    return s.points;
  }
  const xs = s.points.map((p) => p[0]);
  const ys = s.points.map((p) => p[1]);
  const x1 = Math.min(...xs);
  const y1 = Math.min(...ys);
  const x2 = Math.max(...xs);
  const y2 = Math.max(...ys);
  return [
    [x1, y1],
    [x2, y1],
    [x2, y2],
    [x1, y2],
  ];
}

export default function ModelPlayground() {
  const [models, setModels] = useState<api.ModelInfo[]>([]);
  const [loaded, setLoaded] = useState<api.LoadedModelInfo | null>(null);
  const [selectedCfg, setSelectedCfg] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [conf, setConf] = useState(0.25);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);
  const [displayWidth, setDisplayWidth] = useState(0);
  const [shapes, setShapes] = useState<Shape[]>([]);
  const [resultModel, setResultModel] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);
  const previewUrlRef = useRef<string | null>(null);

  const refreshModels = useCallback(async () => {
    try {
      const d = await api.getModels();
      setModels(d.models);
      setLoaded(d.loaded);
      // 默认选中当前已加载模型
      if (d.loaded) {
        setSelectedCfg(d.loaded.config_file);
      }
    } catch (e) {
      console.error(e);
    }
  }, []);

  useEffect(() => {
    refreshModels();
    return () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, [refreshModels]);

  // 预览图显示宽度变化时重算标签字号（shape 坐标是绝对像素）
  useEffect(() => {
    const onResize = () => {
      if (imgRef.current) setDisplayWidth(imgRef.current.clientWidth);
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const onLoad = useCallback(async () => {
    if (!selectedCfg) return;
    setLoading(true);
    try {
      await api.loadModel(selectedCfg);
      const s = await api.getModelStatus();
      setLoaded(s.loaded);
      message.success("模型加载完成");
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      message.error(`模型加载失败: ${err.response?.data?.detail ?? err.message}`);
    } finally {
      setLoading(false);
    }
  }, [selectedCfg]);

  const runPredict = useCallback(
    async (file: File) => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      const url = URL.createObjectURL(file);
      previewUrlRef.current = url;
      setPreviewUrl(url);
      setShapes([]);
      setResultModel(null);
      setRunning(true);
      try {
        const r = await api.playgroundPredict(file, conf);
        setShapes(r.shapes ?? []);
        setResultModel(r.model?.display_name ?? null);
        if ((r.shapes ?? []).length === 0) {
          message.info(
            "未检测到目标：模型只认识训练时学过的类别；也可尝试把置信度阈值调低一点"
          );
        }
      } catch (e) {
        const err = e as { response?: { data?: { detail?: string } }; message: string };
        message.error(`推理失败: ${err.response?.data?.detail ?? err.message}`);
      } finally {
        setRunning(false);
      }
    },
    [conf]
  );

  const onClear = useCallback(() => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = null;
    setPreviewUrl(null);
    setShapes([]);
    setResultModel(null);
    setNaturalSize(null);
  }, []);

  const beforeUpload = useCallback(
    (file: File) => {
      if (!loaded) {
        message.warning("请先加载模型");
        return false;
      }
      if (!file.type.startsWith("image/")) {
        message.warning("请选择图片文件");
        return false;
      }
      if (file.size > 20 * 1024 * 1024) {
        message.warning("图片不能超过 20MB");
        return false;
      }
      void runPredict(file);
      return false; // 拦截自动上传，只走 playground 推理
    },
    [loaded, runPredict]
  );

  const needLoad = !loaded || loaded.config_file !== selectedCfg;
  // 显示缩放比：shape 坐标是绝对像素，按 自然宽度/显示宽度 换算标签字号
  const scale = naturalSize && displayWidth > 0 ? naturalSize.w / displayWidth : 1;
  const fontSize = 12 * scale;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", gap: 8 }}>
        <Select
          size="small"
          style={{ flex: 1 }}
          showSearch
          placeholder="选择模型"
          value={selectedCfg}
          onChange={setSelectedCfg}
          optionFilterProp="label"
          options={models.map((m) => ({
            value: m.config_file,
            label: `${m.display_name} (${m.type})`,
          }))}
          disabled={loading || running}
          dropdownMatchSelectWidth={360}
        />
        {needLoad && (
          <Button
            size="small"
            type="primary"
            loading={loading}
            disabled={!selectedCfg || running}
            onClick={onLoad}
          >
            加载
          </Button>
        )}
      </div>
      {loaded && !needLoad && (
        <div style={{ fontSize: 12, color: "#52c41a" }}>
          <CheckCircleFilled /> 已加载：{loaded.display_name}
        </div>
      )}
      <div style={{ fontSize: 12, color: "#999" }}>
        也可以用左侧训练产物弹窗里的「试用」按钮直接加载训练产物。
      </div>
      <div>
        <div style={{ fontSize: 11, color: "#888" }}>置信度 {conf.toFixed(2)}</div>
        <Slider min={0.05} max={0.95} step={0.05} value={conf} onChange={setConf} />
      </div>
      <Upload.Dragger
        accept="image/*"
        showUploadList={false}
        beforeUpload={beforeUpload}
        disabled={!loaded || loading || running}
        style={{ padding: "4px 0" }}
      >
        <p style={{ margin: 0 }}>
          <InboxOutlined style={{ fontSize: 24, color: "#1677ff" }} />
        </p>
        <p style={{ margin: 0, fontSize: 12, color: "#666" }}>
          {running ? "推理中…" : "点击或拖拽图片到此处试用"}
        </p>
      </Upload.Dragger>

      {previewUrl && (
        <>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 12, color: "#999" }}>试用图片</span>
            <Button size="small" danger type="text" onClick={onClear}>
              清除图片
            </Button>
          </div>
          <div style={{ position: "relative", alignSelf: "flex-start", maxWidth: "100%" }}>
            <img
              ref={imgRef}
              src={previewUrl}
              alt="预览"
              style={{ maxWidth: "100%", display: "block", border: "1px solid #f0f0f0", borderRadius: 4 }}
              onLoad={(e) => {
                const img = e.currentTarget;
                setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
                setDisplayWidth(img.clientWidth);
              }}
            />
            {naturalSize && (
              <svg
                viewBox={`0 0 ${naturalSize.w} ${naturalSize.h}`}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: "100%",
                  pointerEvents: "none",
                }}
              >
                {shapes.map((s, i) => {
                  const outline = shapeOutline(s);
                  if (outline.length === 0) return null;
                  const pts = outline.map((p) => p.join(",")).join(" ");
                  const x0 = Math.min(...outline.map((p) => p[0]));
                  const y0 = Math.min(...outline.map((p) => p[1]));
                  const tag = s.score != null ? `${s.label} ${Math.round(s.score * 100)}%` : s.label;
                  return (
                    <g key={i}>
                      <polygon
                        points={pts}
                        fill="rgba(22,119,255,0.08)"
                        stroke="#1677ff"
                        strokeWidth={2}
                        vectorEffect="non-scaling-stroke"
                      />
                      <text
                        x={x0 + 2 * scale}
                        y={y0 - 4 * scale > fontSize ? y0 - 4 * scale : y0 + fontSize}
                        fontSize={fontSize}
                        fill="#1677ff"
                        stroke="#ffffff"
                        strokeWidth={fontSize / 4}
                        style={{ paintOrder: "stroke" }}
                      >
                        {tag}
                      </text>
                    </g>
                  );
                })}
              </svg>
            )}
          </div>
          <List
            size="small"
            header={
              <div style={{ padding: 0 }}>
                检测结果（{shapes.length}）
                {resultModel && (
                  <span style={{ color: "#999", fontWeight: 400, marginLeft: 8 }}>
                    模型：{resultModel}
                  </span>
                )}
              </div>
            }
            dataSource={shapes}
            locale={{ emptyText: running ? "推理中…" : "未检测到目标" }}
            renderItem={(s, i) => (
              <List.Item style={{ padding: "4px 0" }}>
                <span style={{ width: 24, color: "#999", flexShrink: 0 }}>{i + 1}</span>
                <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis" }}>
                  {s.label}
                </span>
                <span style={{ color: "#888", marginLeft: 8 }}>
                  {s.score != null ? `${(s.score * 100).toFixed(1)}%` : "-"}
                </span>
              </List.Item>
            )}
          />
        </>
      )}
    </div>
  );
}
