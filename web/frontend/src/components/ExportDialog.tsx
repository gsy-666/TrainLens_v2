import { useCallback, useEffect, useRef, useState } from "react";
import {
  Button,
  Checkbox,
  Input,
  message,
  Modal,
  Progress,
  Select,
} from "antd";
import { FolderOpenOutlined } from "@ant-design/icons";
import DirBrowserModal from "./DirBrowserModal";
import * as api from "../api/client";

const FORMAT_LABELS: Record<string, string> = {
  yolo: "YOLO",
  voc: "Pascal VOC",
  coco: "COCO",
  dota: "DOTA（旋转框）",
  mask: "Mask（语义分割）",
  mot: "MOT（多目标跟踪）",
  odvg: "ODVG（ grounding ）",
};

const MODE_LABELS: Record<string, string> = {
  hbb: "水平框 (hbb)",
  obb: "旋转框 (obb)",
  seg: "实例分割 (seg)",
  rectangle: "目标检测 (rectangle)",
  polygon: "实例分割 (polygon)",
};

interface Props {
  open: boolean;
  onCancel: () => void;
}

export default function ExportDialog({ open, onCancel }: Props) {
  const [formats, setFormats] = useState<Record<string, api.ExportFormatInfo>>({});
  const [format, setFormat] = useState("yolo");
  const [mode, setMode] = useState<string>("hbb");
  const [outputDir, setOutputDir] = useState("");
  const [saveImages, setSaveImages] = useState(false);
  const [skipEmpty, setSkipEmpty] = useState(false);
  const [browseOpen, setBrowseOpen] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<{ current: number; total: number } | null>(null);
  const [result, setResult] = useState<api.ExportResult | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    if (open) {
      setResult(null);
      setProgress(null);
      api
        .getExportFormats()
        .then((d) => setFormats(d.formats))
        .catch(() => undefined);
    } else if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, [open]);

  const modes = formats[format]?.modes ?? [];

  useEffect(() => {
    setMode(formats[format]?.default_mode ?? "");
  }, [format, formats]);

  const stopPolling = () => {
    if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const onStart = useCallback(async () => {
    if (!outputDir.trim()) {
      message.warning("请选择导出目录");
      return;
    }
    setRunning(true);
    setResult(null);
    setProgress({ current: 0, total: 1 });
    try {
      await api.startExport({
        format,
        mode: mode || undefined,
        output_dir: outputDir.trim(),
        save_images: saveImages,
        skip_empty_files: skipEmpty,
      });
      pollRef.current = window.setInterval(async () => {
        try {
          const s = await api.getExportStatus();
          setProgress({ current: s.current, total: s.total || 1 });
          if (!s.running) {
            stopPolling();
            setRunning(false);
            if (s.error) {
              message.error(`导出失败: ${s.error}`);
            } else if (s.result) {
              setResult(s.result);
            }
          }
        } catch {
          stopPolling();
          setRunning(false);
        }
      }, 500);
    } catch (e) {
      setRunning(false);
      setProgress(null);
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      message.error(`导出失败: ${err.response?.data?.detail ?? err.message}`);
    }
  }, [format, mode, outputDir, saveImages, skipEmpty]);

  const percent =
    progress && progress.total > 0
      ? Math.round((progress.current / progress.total) * 100)
      : 0;

  return (
    <Modal
      open={open}
      title="导出标注"
      onCancel={onCancel}
      footer={[
        <Button key="cancel" onClick={onCancel}>
          关闭
        </Button>,
        <Button
          key="run"
          type="primary"
          loading={running}
          onClick={onStart}
          disabled={!outputDir.trim()}
        >
          开始导出
        </Button>,
      ]}
      width={480}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 8 }}>
        <div>
          <div style={{ marginBottom: 4 }}>导出格式</div>
          <Select
            style={{ width: "100%" }}
            value={format}
            onChange={setFormat}
            disabled={running}
            options={Object.keys(formats).map((f) => ({
              value: f,
              label: FORMAT_LABELS[f] ?? f,
            }))}
          />
        </div>

        {modes.length > 0 && (
          <div>
            <div style={{ marginBottom: 4 }}>模式</div>
            <Select
              style={{ width: "100%" }}
              value={mode}
              onChange={setMode}
              disabled={running}
              options={modes.map((m) => ({ value: m, label: MODE_LABELS[m] ?? m }))}
            />
          </div>
        )}

        <div>
          <div style={{ marginBottom: 4 }}>导出目录（必须为空目录或不存在）</div>
          <div style={{ display: "flex", gap: 8 }}>
            <Input
              value={outputDir}
              onChange={(e) => setOutputDir(e.target.value)}
              placeholder="选择导出目录"
              disabled={running}
            />
            <Button
              icon={<FolderOpenOutlined />}
              onClick={() => setBrowseOpen(true)}
              disabled={running}
            >
              浏览…
            </Button>
          </div>
        </div>

        <div style={{ display: "flex", gap: 16 }}>
          <Checkbox
            checked={saveImages}
            onChange={(e) => setSaveImages(e.target.checked)}
            disabled={running}
          >
            同时复制图片
          </Checkbox>
          <Checkbox
            checked={skipEmpty}
            onChange={(e) => setSkipEmpty(e.target.checked)}
            disabled={running}
          >
            跳过无标注文件
          </Checkbox>
        </div>

        <div style={{ fontSize: 12, color: "#999" }}>
          类别列表自动从当前数据集的标注中提取。
        </div>

        {running && <Progress percent={percent} size="small" status="active" />}

        {result && (
          <div
            style={{
              background: "#f6ffed",
              border: "1px solid #b7eb8f",
              borderRadius: 6,
              padding: 12,
            }}
          >
            <div>导出完成：{result.files_written} 个文件</div>
            <div style={{ fontSize: 12, color: "#666", wordBreak: "break-all", margin: "4px 0 8px" }}>
              {result.output_dir}
            </div>
            <Button
              size="small"
              type="primary"
              href={api.exportDownloadUrl(result.output_dir)}
              target="_blank"
            >
              下载 ZIP
            </Button>
          </div>
        )}
      </div>

      <DirBrowserModal
        open={browseOpen}
        onCancel={() => setBrowseOpen(false)}
        onSelect={(p) => {
          setOutputDir(p);
          setBrowseOpen(false);
        }}
      />
    </Modal>
  );
}
