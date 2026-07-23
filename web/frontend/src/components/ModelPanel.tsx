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
} from "@ant-design/icons";
import * as api from "../api/client";
import { useStudio } from "../store/useStudio";
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

  const refreshModels = useCallback(async () => {
    try {
      const d = await api.getModels();
      setModels(d.models);
      setLoaded(d.loaded);
      // keep the dropdown in sync with the actually loaded model
      if (d.loaded) setSelectedCfg(d.loaded.config_file);
    } catch (e) {
      console.error(e);
    }
  }, []);

  useEffect(() => {
    refreshModels();
    // pick up a pending batch-undo state from an earlier run
    api
      .getBatchStatus()
      .then((s) => setUndoCount(s.undo_available ? s.backup_count ?? 0 : 0))
      .catch(() => undefined);
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, [refreshModels]);

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

  const onLoad = useCallback(async () => {
    if (!selectedCfg) return;
    setLoading(true);
    setProgress(null);
    setStatusMsg("开始加载模型...");
    startStatusPolling((s) => !s.loading && s.loaded !== null);
    // fire and poll; the POST resolves when loading finished
    api
      .loadModel(selectedCfg)
      .then(async () => {
        const s = await api.getModelStatus();
        setLoaded(s.loaded);
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
  }, [selectedCfg, startStatusPolling]);

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
        disabled={loading}
        size="small"
        dropdownMatchSelectWidth={360}
      />

      <Space style={{ marginTop: 8, width: "100%" }} size={4}>
        {!loaded ? (
          <Button
            size="small"
            type="primary"
            onClick={onLoad}
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
          <Input
            size="small"
            placeholder="文本提示：多类用英文句号分隔，如 person. car."
            value={textPrompt}
            onChange={(e) => setTextPrompt(e.target.value)}
            style={{ marginBottom: 6 }}
          />
          <div style={{ fontSize: 11, color: "#888" }}>置信度 {conf.toFixed(2)}</div>
          <Slider min={0.05} max={0.95} step={0.05} value={conf} onChange={setConf} />
          <div style={{ fontSize: 11, color: "#888" }}>IoU {iou.toFixed(2)}</div>
          <Slider min={0.1} max={0.95} step={0.05} value={iou} onChange={setIou} />
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
                  label: `仅未标注（无标签或空标注，${
                    images.filter((im) => !im.has_label || (im.shape_count ?? 0) === 0).length
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
    </div>
  );
}
