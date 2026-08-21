import { useCallback, useEffect, useRef, useState } from "react";
import {
  Button,
  Divider,
  Input,
  InputNumber,
  message,
  Modal,
  Popconfirm,
  Progress,
  Radio,
  Select,
  Slider,
  Space,
  Switch,
  Tag,
  Tooltip,
} from "antd";
import {
  CheckCircleFilled,
  ThunderboltOutlined,
  RobotOutlined,
  RollbackOutlined,
  FolderOpenOutlined,
  DatabaseOutlined,
  DeleteOutlined,
} from "@ant-design/icons";
import * as api from "../api/client";
import { useStudio } from "../store/useStudio";
import DirBrowserModal from "./DirBrowserModal";
import type { Shape } from "../types";

function normalizePredictedShape(s: Shape): Shape {
  return {
    ...s,
    description: s.description ?? "",
    flags: s.flags ?? {},
    attributes: s.attributes ?? {},
    group_id: s.group_id ?? null,
  };
}

export default function ModelPanel() {
  const {
    images,
    video,
    currentIndex,
    shapes,
    setShapesExternal,
    reloadCurrent,
    refreshImages,
    samMode,
    setSamMode,
  } = useStudio();

  const [models, setModels] = useState<api.ModelInfo[]>([]);
  const [loaded, setLoaded] = useState<api.LoadedModelInfo | null>(null);
  const [selectedCfg, setSelectedCfg] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState<{ downloaded: number; total: number } | null>(null);
  const [statusMsg, setStatusMsg] = useState("");
  const [running, setRunning] = useState(false);
  const [batch, setBatch] = useState<api.BatchStatus | null>(null);
  const [textPrompt, setTextPrompt] = useState("");
  const [outputMode, setOutputMode] = useState<string>("rectangle");
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.45);
  const [preserve, setPreserve] = useState(false);
  const [undoCount, setUndoCount] = useState(0);
  const [undoing, setUndoing] = useState(false);
  const [track, setTrack] = useState<api.TrackStatus | null>(null);
  const [batchOpen, setBatchOpen] = useState(false);
  const [batchScope, setBatchScope] = useState<"all" | "unlabeled" | "prev" | "next">("all");
  const [batchN, setBatchN] = useState(100);
  const pollRef = useRef<number | null>(null);

  // local weight registration
  const [localOpen, setLocalOpen] = useState(false);
  const [localTemplate, setLocalTemplate] = useState<string | undefined>();
  const [localPath, setLocalPath] = useState("");
  const [localName, setLocalName] = useState("");
  const [localBusy, setLocalBusy] = useState(false);
  const [localBrowseOpen, setLocalBrowseOpen] = useState(false);

  // local model library (downloaded cache scan)
  const [localFiles, setLocalFiles] = useState<Record<string, api.LocalModelFileInfo>>({});
  const [localFilesRoot, setLocalFilesRoot] = useState("");
  const [localFilesTotal, setLocalFilesTotal] = useState(0);
  const [libOpen, setLibOpen] = useState(false);
  const [scanPath, setScanPath] = useState("");
  const [scanFiles, setScanFiles] = useState<api.ScannedModelFile[] | null>(null);
  const [scanBusy, setScanBusy] = useState(false);
  const [libBrowseOpen, setLibBrowseOpen] = useState(false);

  // device preference
  const [inferenceDevice, setInferenceDevice] = useState<string>("auto");
  const [availableDevices, setAvailableDevices] = useState<string[]>(["CPU"]);

  const refreshModels = useCallback(async () => {
    try {
      const d = await api.getModels();
      setModels(d.models);
      setLoaded(d.loaded);
      // keep the dropdown in sync with the actually loaded model
      if (d.loaded) {
        setSelectedCfg(d.loaded.config_file);
        setOutputMode(d.loaded.default_output_mode ?? "rectangle");
      }
    } catch (e) {
      console.error(e);
    }
  }, []);

  useEffect(() => {
    refreshModels();
    // fetch device preference
    api.getInferenceDevice().then((d) => {
      setInferenceDevice(d.current === "auto" ? "auto" : d.current);
      setAvailableDevices(d.available);
    }).catch(() => undefined);
    // pick up a pending batch-undo state from an earlier run
    api
      .getBatchStatus()
      .then((s) => setUndoCount(s.undo_available ? s.backup_count ?? 0 : 0))
      .catch(() => undefined);
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, [refreshModels]);

  const refreshLocalFiles = useCallback(async () => {
    try {
      const d = await api.getLocalModelFiles();
      const map: Record<string, api.LocalModelFileInfo> = {};
      for (const it of d.items) map[it.config_file] = it;
      setLocalFiles(map);
      setLocalFilesRoot(d.root);
      setLocalFilesTotal(d.total_bytes);
    } catch {
      /* scan is best-effort */
    }
  }, []);

  useEffect(() => {
    refreshLocalFiles();
  }, [refreshLocalFiles]);

  const onDeleteCache = useCallback(
    async (cfg: string) => {
      try {
        const r = await api.deleteModelCache(cfg);
        message.success(
          r.deleted ? `已删除缓存，释放 ${(r.freed_bytes / 1024 / 1024).toFixed(1)} MB` : "缓存文件不存在"
        );
        await refreshLocalFiles();
      } catch (e) {
        const err = e as { response?: { data?: { detail?: string } }; message: string };
        message.error(`删除失败: ${err.response?.data?.detail ?? err.message}`);
      }
    },
    [refreshLocalFiles]
  );

  const onScanDir = useCallback(async () => {
    if (!scanPath.trim()) return;
    setScanBusy(true);
    try {
      const d = await api.scanModelDir(scanPath.trim());
      setScanFiles(d.files);
      if (d.files.length === 0) message.info("该目录下没有找到 .onnx 文件");
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      message.error(`扫描失败: ${err.response?.data?.detail ?? err.message}`);
    } finally {
      setScanBusy(false);
    }
  }, [scanPath]);

  const startStatusPolling = useCallback(
    (until: (s: api.ModelStatus) => boolean) => {
      if (pollRef.current) window.clearInterval(pollRef.current);
      pollRef.current = window.setInterval(async () => {
        try {
          const s = await api.getModelStatus();
          setProgress(s.progress);
          setStatusMsg(s.message);
          if (until(s)) {
            if (pollRef.current) window.clearInterval(pollRef.current);
            pollRef.current = null;
          }
        } catch {
          /* ignore */
        }
      }, 800);
    },
    []
  );

  const onDeviceChange = useCallback(async (device: string) => {
    setInferenceDevice(device);
    try {
      await api.setInferenceDevice(device);
      if (loaded) {
        message.info("设备已切换，重新加载模型后生效");
      }
    } catch (e) {
      message.error(`切换设备失败: ${(e as Error).message}`);
    }
  }, [loaded]);

  const onLoad = useCallback(async (cfg?: string) => {
    const target = cfg ?? selectedCfg;
    if (!target) return;
    setLoading(true);
    setProgress(null);
    setStatusMsg("开始加载模型...");
    startStatusPolling((s) => !s.loading && s.loaded !== null);
    // fire and poll; the POST resolves when loading finished
    api
      .loadModel(target)
      .then(async () => {
        const s = await api.getModelStatus();
        setLoaded(s.loaded);
        setSelectedCfg(s.loaded?.config_file ?? target);
        setOutputMode(s.loaded?.default_output_mode ?? "rectangle");
        refreshLocalFiles(); // a download may have completed
        message.success("模型加载完成");
      })
      .catch((e) => {
        message.error(`模型加载失败: ${e?.response?.data?.detail ?? e.message}`);
      })
      .finally(() => {
        setLoading(false);
        setProgress(null);
        if (pollRef.current) window.clearInterval(pollRef.current);
      });
  }, [selectedCfg, startStatusPolling, refreshLocalFiles]);

  const onRegisterLocal = useCallback(async () => {
    if (!localTemplate || !localPath.trim()) return;
    setLocalBusy(true);
    try {
      const r = await api.registerLocalModel({
        template_config_file: localTemplate,
        local_path: localPath.trim(),
        display_name: localName.trim() || undefined,
      });
      message.success(`已注册本地权重: ${r.display_name}`);
      setLocalOpen(false);
      setLocalPath("");
      setLocalName("");
      await refreshModels();
      await refreshLocalFiles();
      setSelectedCfg(r.config_file);
      onLoad(r.config_file);
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      message.error(`注册失败: ${err.response?.data?.detail ?? err.message}`);
    } finally {
      setLocalBusy(false);
    }
  }, [localTemplate, localPath, localName, refreshModels, refreshLocalFiles, onLoad]);

  const onUnload = useCallback(async () => {
    await api.unloadModel();
    setLoaded(null);
    setSamMode(false);
  }, [setSamMode]);

  const onRunCurrent = useCallback(async () => {
    if (currentIndex < 0 || !loaded) return;
    setRunning(true);
    const hide = message.loading("推理中，请稍候…", 0);
    const t0 = performance.now();
    try {
      const res = await api.predict(
        images[currentIndex].filename,
        textPrompt || undefined,
        conf,
        iou
      );
      const predicted = res.shapes.map(normalizePredictedShape);
      if (res.replace) {
        setShapesExternal(predicted);
      } else {
        setShapesExternal([...shapes, ...predicted]);
      }
      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      if (predicted.length === 0) {
        message.warning(
          loaded?.type === "segment_anything_3" && !textPrompt.trim()
            ? "未分割出目标：SAM3 是文本引导模型，请先在文本提示框填写类别（如 gauge.）"
            : `未检测到目标（${secs}s）。可尝试降低置信度阈值，或检查文本提示`,
          5
        );
      } else {
        message.success(`推理完成：${predicted.length} 个形状（${secs}s）`);
      }
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      message.error(`推理失败: ${err.response?.data?.detail ?? err.message}`);
    } finally {
      hide();
      setRunning(false);
    }
  }, [currentIndex, loaded, images, textPrompt, conf, iou, shapes, setShapesExternal]);

  // compute the image slice for the chosen batch scope
  const batchTargets = useCallback((): string[] => {
    if (batchScope === "unlabeled") {
      return images
        .filter((im) => !im.has_label || (im.shape_count ?? 0) === 0)
        .map((im) => im.filename);
    }
    if (batchScope === "prev") {
      const start = Math.max(0, currentIndex - batchN);
      return images.slice(start, currentIndex).map((im) => im.filename);
    }
    if (batchScope === "next") {
      return images
        .slice(currentIndex + 1, currentIndex + 1 + batchN)
        .map((im) => im.filename);
    }
    return images.map((im) => im.filename);
  }, [images, batchScope, batchN, currentIndex]);

  const onRunBatch = useCallback(async () => {
    const targets = batchTargets();
    if (targets.length === 0) {
      message.info("范围内没有需要处理的图片");
      return;
    }
    setBatchOpen(false);
    setBatch({ running: true, current: 0, total: targets.length });
    try {
      await api.predictBatch(targets, preserve, conf, iou, textPrompt || undefined);
      const timer = window.setInterval(async () => {
        const s = await api.getBatchStatus();
        setBatch(s);
        if (!s.running) {
          window.clearInterval(timer);
          setUndoCount(s.undo_available ? s.backup_count ?? 0 : 0);
          await refreshImages();
          await reloadCurrent();
          const errs = s.errors?.length ?? 0;
          if (errs > 0) {
            message.warning(`批量预标注完成，${errs} 张失败`);
          } else {
            message.success("批量预标注完成");
          }
        }
      }, 1000);
    } catch (e) {
      setBatch(null);
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      message.error(`批量任务失败: ${err.response?.data?.detail ?? err.message}`);
    }
  }, [batchTargets, preserve, conf, iou, textPrompt, refreshImages, reloadCurrent]);

  const onUndoBatch = useCallback(async () => {
    setUndoing(true);
    try {
      const r = await api.undoBatch();
      const skipped = r.skipped_modified.length;
      message.success(
        `已撤回：恢复 ${r.restored} 个、删除 ${r.deleted} 个新生成的标注` +
        (skipped ? `，${skipped} 个因之后被手动修改而跳过` : "")
      );
      setUndoCount(0);
      await refreshImages();
      await reloadCurrent();
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      message.error(`撤回失败: ${err.response?.data?.detail ?? err.message}`);
    } finally {
      setUndoing(false);
    }
  }, [refreshImages, reloadCurrent]);

  const onRunTrack = useCallback(async () => {
    setTrack({ running: true, current: 0, total: video?.frameCount ?? 1, current_frame: null, errors: [], result: null });
    try {
      await api.startTrack({ conf, iou, preserve_existing: preserve });
      const timer = window.setInterval(async () => {
        try {
          const s = await api.getTrackStatus();
          setTrack(s);
          if (!s.running) {
            window.clearInterval(timer);
            setUndoCount(s.undo_available ? s.total : 0);
            await refreshImages();
            await reloadCurrent();
            const errs = s.errors?.length ?? 0;
            if (errs > 0) {
              message.warning(`跟踪完成，${errs} 帧失败`);
            } else {
              message.success(`跟踪完成：${s.total} 帧`);
            }
          }
        } catch {
          window.clearInterval(timer);
        }
      }, 1000);
    } catch (e) {
      setTrack(null);
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      message.error(`跟踪失败: ${err.response?.data?.detail ?? err.message}`);
    }
  }, [video, conf, iou, preserve, refreshImages, reloadCurrent]);

  const downloadPercent =
    progress && progress.total > 0
      ? Math.round((progress.downloaded / progress.total) * 100)
      : null;

  return (
    <div style={{ padding: 12, borderTop: "1px solid #f0f0f0" }}>
      <div style={{ fontWeight: 600, marginBottom: 8 }}>
        <RobotOutlined /> 自动标注
        {loaded && (
          <Tag color="green" style={{ marginLeft: 8 }}>
            <CheckCircleFilled /> {loaded.display_name}
          </Tag>
        )}
      </div>

      <Select
        style={{ width: "100%" }}
        showSearch
        placeholder="选择模型"
        value={selectedCfg}
        onChange={setSelectedCfg}
        optionFilterProp="label"
        options={models.map((m) => ({
          value: m.config_file,
          label: `${m.display_name} (${m.type})`,
        }))}
        optionRender={(info) => {
          const cfg = String(info.data.value);
          const lf = localFiles[cfg];
          const m = models.find((x) => x.config_file === cfg);
          return (
            <span>
              {info.data.label}
              {lf?.downloaded && (
                <Tag color="green" style={{ marginLeft: 6 }}>
                  已下载
                </Tag>
              )}
              {m?.is_custom_model && <Tag style={{ marginLeft: 4 }}>自定义</Tag>}
            </span>
          );
        }}
        disabled={loading}
        size="small"
        dropdownMatchSelectWidth={360}
      />
      <div style={{ marginTop: 4, textAlign: "right" }}>
        {selectedCfg && models.find((m) => m.config_file === selectedCfg)?.is_custom_model && (
          <Popconfirm
            title="从列表移除该自定义模型？"
            description="不会删除权重文件本身"
            okText="移除"
            cancelText="取消"
            onConfirm={async () => {
              try {
                await api.deleteCustomModel(selectedCfg, true);
                message.success("已移除自定义模型");
                if (loaded?.config_file === selectedCfg) {
                  await api.unloadModel();
                  setLoaded(null);
                }
                setSelectedCfg(undefined);
                await refreshModels();
              } catch (e) {
                const err = e as { response?: { data?: { detail?: string } }; message: string };
                message.error(`移除失败: ${err.response?.data?.detail ?? err.message}`);
              }
            }}
          >
            <Button type="link" size="small" danger style={{ padding: 0, fontSize: 12 }}>
              移除
            </Button>
          </Popconfirm>
        )}
        <Button
          type="link"
          size="small"
          style={{ padding: 0, fontSize: 12 }}
          icon={<DatabaseOutlined />}
          onClick={() => {
            refreshLocalFiles();
            setLibOpen(true);
          }}
        >
          模型库
        </Button>
        <Button
          type="link"
          size="small"
          style={{ padding: 0, fontSize: 12 }}
          icon={<FolderOpenOutlined />}
          onClick={() => {
            setLocalTemplate(selectedCfg);
            setLocalOpen(true);
          }}
        >
          使用本地权重文件…
        </Button>
      </div>

      {availableDevices.length > 1 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 11, color: "#888", marginBottom: 4 }}>推理设备</div>
          <Radio.Group
            size="small"
            value={inferenceDevice}
            onChange={(e) => onDeviceChange(e.target.value)}
            optionType="button"
            buttonStyle="solid"
          >
            <Radio.Button value="auto">自动</Radio.Button>
            <Radio.Button value="gpu">
              <span style={{ fontSize: 11 }}>GPU</span>
            </Radio.Button>
            <Radio.Button value="cpu">
              <span style={{ fontSize: 11 }}>CPU</span>
            </Radio.Button>
          </Radio.Group>
        </div>
      )}

      <Space style={{ marginTop: 8, width: "100%" }} size={4}>
        {!loaded ? (
          <Button
            size="small"
            type="primary"
            onClick={() => onLoad()}
            loading={loading}
            disabled={!selectedCfg}
          >
            加载模型
          </Button>
        ) : (
          <Button size="small" onClick={onUnload}>
            卸载模型
          </Button>
        )}
        {!video && (
          <Tooltip title="对当前图片运行推理">
            <Button
              size="small"
              type="primary"
              ghost
              icon={<ThunderboltOutlined />}
              disabled={!loaded || running || currentIndex < 0}
              loading={running}
              onClick={onRunCurrent}
            >
              运行
            </Button>
          </Tooltip>
        )}
        {!video && (
          <Button
            size="small"
            disabled={!loaded || images.length === 0 || !!batch?.running}
            onClick={() => setBatchOpen(true)}
          >
            批量
          </Button>
        )}
        {video && (
          <Popconfirm
            title={`对全部 ${video.frameCount} 帧运行 MOT 跟踪？`}
            description="需加载 bytetrack/botsort/tracktrack 类模型"
            onConfirm={onRunTrack}
          >
            <Button
              size="small"
              type="primary"
              ghost
              icon={<ThunderboltOutlined />}
              disabled={!loaded || !!track?.running}
            >
              跟踪
            </Button>
          </Popconfirm>
        )}
        {undoCount > 0 && !batch?.running && !track?.running && (
          <Popconfirm
            title={`恢复到自动标注前的状态？`}
            description="之后被手动修改的文件会跳过"
            onConfirm={onUndoBatch}
          >
            <Button size="small" danger icon={<RollbackOutlined />} loading={undoing}>
              撤回
            </Button>
          </Popconfirm>
        )}
      </Space>

      {loading && (
        <div style={{ marginTop: 8 }}>
          {downloadPercent !== null ? (
            <Progress percent={downloadPercent} size="small" />
          ) : (
            <Progress percent={100} status="active" size="small" showInfo={false} />
          )}
          <div style={{ fontSize: 11, color: "#999", marginTop: 2 }}>{statusMsg}</div>
        </div>
      )}

      {batch?.running && (
        <div style={{ marginTop: 8 }}>
          <Progress
            percent={Math.round(((batch.current ?? 0) / (batch.total ?? 1)) * 100)}
            size="small"
          />
          <div style={{ fontSize: 11, color: "#999" }}>{batch.current_image}</div>
        </div>
      )}

      {track?.running && (
        <div style={{ marginTop: 8 }}>
          <Progress
            percent={Math.round((track.current / (track.total || 1)) * 100)}
            size="small"
          />
          <div style={{ fontSize: 11, color: "#999" }}>
            跟踪中：第 {(track.current_frame ?? 0) + 1} 帧
          </div>
        </div>
      )}

      {loaded && (
        <div style={{ marginTop: 8 }}>
          {/* 输出形状：模型支持多种输出模式时才显示 */}
          {(loaded.output_modes?.length ?? 0) > 1 && (
            <div style={{ marginBottom: 6 }}>
              <div style={{ fontSize: 11, color: "#888" }}>输出形状</div>
              <Select
                size="small"
                style={{ width: "100%" }}
                value={outputMode}
                onChange={async (v) => {
                  setOutputMode(v);
                  try {
                    await api.setOutputMode(v);
                  } catch (e) {
                    message.error(`切换输出模式失败: ${(e as Error).message}`);
                  }
                }}
                options={(loaded.output_modes ?? []).map((m) => ({
                  value: m,
                  label:
                    m === "rectangle"
                      ? "矩形"
                      : m === "polygon"
                        ? "多边形"
                        : m === "rotation"
                          ? "旋转框"
                          : m === "point"
                            ? "点"
                            : m,
                }))}
              />
            </div>
          )}
          {/* 文本提示：edit_text 类模型（grounding/SAM3 等） */}
          {(loaded.widgets ?? []).includes("edit_text") && (
            <Input
              size="small"
              placeholder="文本提示：多类用英文句号分隔，如 person. car."
              value={textPrompt}
              onChange={(e) => setTextPrompt(e.target.value)}
              style={{ marginBottom: 6 }}
            />
          )}
          {/* 置信度：edit_conf / input_conf / conf_threshold 类模型 */}
          {["edit_conf", "input_conf", "conf_threshold"].some((w) =>
            (loaded.widgets ?? []).includes(w)
          ) && (
              <>
                <div style={{ fontSize: 11, color: "#888" }}>置信度 {conf.toFixed(2)}</div>
                <Slider min={0.05} max={0.95} step={0.05} value={conf} onChange={setConf} />
              </>
            )}
          {/* IoU：edit_iou / input_iou 类模型 */}
          {["edit_iou", "input_iou"].some((w) => (loaded.widgets ?? []).includes(w)) && (
            <>
              <div style={{ fontSize: 11, color: "#888" }}>IoU {iou.toFixed(2)}</div>
              <Slider min={0.1} max={0.95} step={0.05} value={iou} onChange={setIou} />
            </>
          )}
          <div style={{ fontSize: 11, color: "#888" }}>
            保留已有标注{" "}
            <Switch size="small" checked={preserve} onChange={setPreserve} />
          </div>
          {loaded.supports_marks && !video && (
            <div style={{ fontSize: 11, color: "#888", marginTop: 4 }}>
              SAM 交互（画布上点/框提示）{" "}
              <Switch
                size="small"
                checked={samMode}
                onChange={(v) => {
                  setSamMode(v);
                  if (v) message.info("SAM 模式：单击出点，拖拽出框", 3);
                }}
              />
            </div>
          )}
        </div>
      )}

      <Divider style={{ margin: "8px 0 0" }} />

      <Modal
        open={batchOpen}
        title="批量预标注"
        okText="开始"
        cancelText="取消"
        onCancel={() => setBatchOpen(false)}
        onOk={onRunBatch}
        width={420}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 14, marginTop: 8 }}>
          <div>
            <div style={{ marginBottom: 6 }}>范围</div>
            <Radio.Group
              value={batchScope}
              onChange={(e) => setBatchScope(e.target.value)}
              style={{ display: "flex", flexDirection: "column", gap: 8 }}
              options={[
                { value: "all", label: `全部（${images.length} 张）` },
                {
                  value: "unlabeled",
                  label: `仅未标注（无标签或空标注，${images.filter((im) => !im.has_label || (im.shape_count ?? 0) === 0).length
                    } 张）`,
                },
                {
                  value: "prev",
                  label: (
                    <span>
                      当前向前（不含当前）{" "}
                      <InputNumber
                        size="small"
                        min={1}
                        max={images.length}
                        value={batchN}
                        onChange={(v) => setBatchN(v ?? 1)}
                        style={{ width: 80 }}
                        onClick={(e) => e.stopPropagation()}
                      />{" "}
                      张
                    </span>
                  ),
                },
                {
                  value: "next",
                  label: (
                    <span>
                      当前向后（不含当前）{" "}
                      <InputNumber
                        size="small"
                        min={1}
                        max={images.length}
                        value={batchN}
                        onChange={(v) => setBatchN(v ?? 1)}
                        style={{ width: 80 }}
                        onClick={(e) => e.stopPropagation()}
                      />{" "}
                      张
                    </span>
                  ),
                },
              ]}
            />
          </div>
          <div style={{ fontSize: 12, color: "#888" }}>
            将处理 {batchTargets().length} 张图片
            {currentIndex >= 0 && batchScope !== "all" && batchScope !== "unlabeled" && (
              <>，以当前第 {currentIndex + 1} 张为基准</>
            )}
            。大数量任务可随时用「撤回」整批恢复。
          </div>
        </div>
      </Modal>

      <Modal
        open={localOpen}
        title="使用本地权重文件"
        okText="注册并加载"
        cancelText="取消"
        confirmLoading={localBusy}
        onCancel={() => setLocalOpen(false)}
        onOk={onRegisterLocal}
        okButtonProps={{ disabled: !localTemplate || !localPath.trim() }}
        width={440}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 8 }}>
          <div style={{ fontSize: 12, color: "#71717a" }}>
            已手动下载好的 .onnx 权重可直接注册使用，跳过自动下载。模型配置（输入尺寸 / 阈值 / 类别）继承所选模板模型。
          </div>
          <div>
            <div style={{ marginBottom: 4 }}>模板模型</div>
            <Select
              style={{ width: "100%" }}
              showSearch
              optionFilterProp="label"
              placeholder="选择与该权重结构一致的模型"
              value={localTemplate}
              onChange={setLocalTemplate}
              options={models.map((m) => ({
                value: m.config_file,
                label: `${m.display_name} (${m.type})`,
              }))}
            />
          </div>
          <div>
            <div style={{ marginBottom: 4 }}>本地权重文件（.onnx）</div>
            <Space.Compact style={{ width: "100%" }}>
              <Input
                placeholder="例如 D:/models/yolov8n.onnx"
                value={localPath}
                onChange={(e) => setLocalPath(e.target.value)}
              />
              <Button icon={<FolderOpenOutlined />} onClick={() => setLocalBrowseOpen(true)}>
                浏览
              </Button>
            </Space.Compact>
          </div>
          <div>
            <div style={{ marginBottom: 4 }}>显示名称（可选）</div>
            <Input
              placeholder="默认为「模板名(本地 文件名)」"
              value={localName}
              onChange={(e) => setLocalName(e.target.value)}
              maxLength={60}
            />
          </div>
        </div>
      </Modal>

      <DirBrowserModal
        open={localBrowseOpen}
        title="选择 ONNX 权重文件"
        fileExtensions={[".onnx"]}
        onSelect={(p) => {
          setLocalPath(p);
          setLocalBrowseOpen(false);
        }}
        onCancel={() => setLocalBrowseOpen(false)}
      />

      <Modal
        open={libOpen}
        title="模型库"
        footer={null}
        onCancel={() => setLibOpen(false)}
        width={640}
      >
        {(() => {
          const downloaded = Object.values(localFiles).filter((i) => i.downloaded);
          const brokenCustom = Object.values(localFiles).filter(
            (i) => i.is_custom_model && !i.downloaded
          );
          const fmtMB = (n: number) => `${(n / 1024 / 1024).toFixed(1)} MB`;
          return (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div style={{ fontSize: 12, color: "#71717a" }}>
                缓存目录:{localFilesRoot} · 已下载 {downloaded.length} 个模型,共占用{" "}
                {fmtMB(localFilesTotal)}。选择模型点击「加载模型」时会自动下载缺失的权重。
              </div>

              {downloaded.length === 0 ? (
                <div style={{ fontSize: 12, color: "#a1a1aa" }}>暂无已下载模型</div>
              ) : (
                <div style={{ maxHeight: 260, overflow: "auto" }}>
                  {downloaded.map((it) => (
                    <div
                      key={it.config_file}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        padding: "6px 0",
                        borderBottom: "1px solid #f4f4f5",
                      }}
                    >
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <span style={{ fontSize: 13 }}>{it.display_name}</span>{" "}
                        <Tag style={{ fontSize: 11 }}>{it.type}</Tag>
                        {it.is_custom_model && <Tag style={{ fontSize: 11 }}>自定义</Tag>}
                        <div
                          style={{
                            fontSize: 11,
                            color: "#a1a1aa",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                          }}
                          title={it.path ?? undefined}
                        >
                          {it.path}
                        </div>
                      </div>
                      <span style={{ fontSize: 12, color: "#71717a" }}>
                        {fmtMB(it.size_bytes)}
                      </span>
                      {loaded?.config_file === it.config_file ? (
                        <Tag color="green" style={{ marginRight: 0 }}>
                          已加载
                        </Tag>
                      ) : (
                        <Tooltip title="直接加载该模型">
                          <Button
                            size="small"
                            type="primary"
                            ghost
                            disabled={loading}
                            onClick={() => {
                              setLibOpen(false);
                              setSelectedCfg(it.config_file);
                              onLoad(it.config_file);
                            }}
                          >
                            加载
                          </Button>
                        </Tooltip>
                      )}
                      {!it.is_custom_model && (
                        <Popconfirm
                          title="删除该模型缓存?"
                          description="下次加载时会重新下载"
                          okText="删除"
                          cancelText="取消"
                          onConfirm={() => onDeleteCache(it.config_file)}
                        >
                          <Button size="small" type="text" danger icon={<DeleteOutlined />} />
                        </Popconfirm>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {brokenCustom.length > 0 && (
                <div style={{ fontSize: 12, color: "#d97706" }}>
                  {brokenCustom.length} 个自定义模型的权重文件已不在原路径,请重新注册或移除。
                </div>
              )}

              <div style={{ borderTop: "1px solid #f0f0f0", paddingTop: 12 }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>
                  扫描目录发现权重
                </div>
                <Space.Compact style={{ width: "100%" }}>
                  <Input
                    placeholder="选择或输入目录,发现其中的 .onnx 文件"
                    value={scanPath}
                    onChange={(e) => setScanPath(e.target.value)}
                    onPressEnter={onScanDir}
                  />
                  <Button icon={<FolderOpenOutlined />} onClick={() => setLibBrowseOpen(true)}>
                    浏览
                  </Button>
                  <Button type="primary" onClick={onScanDir} loading={scanBusy}>
                    扫描
                  </Button>
                </Space.Compact>
                {scanFiles && scanFiles.length > 0 && (
                  <div style={{ maxHeight: 180, overflow: "auto", marginTop: 8 }}>
                    {scanFiles.map((f) => (
                      <div
                        key={f.path}
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 8,
                          padding: "4px 0",
                          borderBottom: "1px solid #f4f4f5",
                        }}
                      >
                        <span style={{ flex: 1, fontSize: 12, overflow: "hidden", textOverflow: "ellipsis" }} title={f.path}>
                          {f.name}
                        </span>
                        <span style={{ fontSize: 11, color: "#a1a1aa" }}>{fmtMB(f.size_bytes)}</span>
                        <Button
                          size="small"
                          type="link"
                          onClick={() => {
                            setLocalPath(f.path);
                            setLibOpen(false);
                            setLocalOpen(true);
                          }}
                        >
                          注册
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        })()}
      </Modal>

      <DirBrowserModal
        open={libBrowseOpen}
        title="选择要扫描的目录"
        onSelect={(p) => {
          setScanPath(p);
          setLibBrowseOpen(false);
        }}
        onCancel={() => setLibBrowseOpen(false)}
      />
    </div>
  );
}
