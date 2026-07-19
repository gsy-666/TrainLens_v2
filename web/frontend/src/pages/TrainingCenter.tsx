import { useCallback, useEffect, useRef, useState } from "react";
import {
  Button,
  Card,
  Collapse,
  Input,
  InputNumber,
  List,
  message,
  Modal,
  Select,
  Slider,
  Space,
  Table,
  Tag,
  Tooltip,
} from "antd";
import {
  ArrowLeftOutlined,
  CaretRightOutlined,
  CheckCircleFilled,
  CloseCircleFilled,
  ExperimentOutlined,
  FolderOpenOutlined,
  StopOutlined,
} from "@ant-design/icons";
import * as api from "../api/client";
import DirBrowserModal from "../components/DirBrowserModal";
import LineChart from "../components/LineChart";

const SEV_STYLE: Record<string, { color: string; icon: React.ReactNode }> = {
  pass: { color: "#52c41a", icon: <CheckCircleFilled /> },
  warning: { color: "#faad14", icon: <CloseCircleFilled /> },
  error: { color: "#f5222d", icon: <CloseCircleFilled /> },
};

interface Props {
  onBack: () => void;
}

export default function TrainingCenter({ onBack }: Props) {
  // form
  const [task, setTask] = useState("detect");
  const [model, setModel] = useState("yolov8n.pt");
  const [data, setData] = useState("");
  const [project, setProject] = useState("");
  const [name, setName] = useState("train");
  const [device, setDevice] = useState("cpu");
  const [epochs, setEpochs] = useState(100);
  const [batch, setBatch] = useState(16);
  const [imgsz, setImgsz] = useState(640);
  const [patience, setPatience] = useState<number | null>(null);
  const [lr0, setLr0] = useState<number | null>(null);
  const [browse, setBrowse] = useState<"data" | "project" | null>(null);
  const [prepOpen, setPrepOpen] = useState(false);
  const [prepTask, setPrepTask] = useState("Detect");
  const [prepRatio, setPrepRatio] = useState(0.9);
  const [preparing, setPreparing] = useState(false);

  // runtime
  const [issues, setIssues] = useState<api.PreflightIssue[] | null>(null);
  const [checking, setChecking] = useState(false);
  const [status, setStatus] = useState<api.TrainingStatusResponse | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<api.MetricSeries[]>([]);
  const [history, setHistory] = useState<Record<string, unknown>[]>([]);

  const seqRef = useRef(0);
  const logBoxRef = useRef<HTMLDivElement>(null);
  const running = !!status?.running;

  const formPayload = useCallback(
    () => ({
      task,
      model,
      data,
      project,
      name,
      device,
      epochs,
      batch,
      imgsz,
      ...(patience != null ? { patience } : {}),
      ...(lr0 != null ? { lr0 } : {}),
    }),
    [task, model, data, project, name, device, epochs, batch, imgsz, patience, lr0]
  );

  // ---- polling -------------------------------------------------------------
  useEffect(() => {
    let stopped = false;

    const pollEvents = async () => {
      try {
        const d = await api.getTrainingEvents(seqRef.current);
        seqRef.current = d.latest;
        if (d.events.length > 0) {
          setLogs((prev) => {
            const lines = d.events
              .filter((e) => e.event_type === "console_output")
              .map((e) => String((e.payload as { message?: string })?.message ?? ""));
            const others = d.events
              .filter((e) => e.event_type !== "console_output")
              .map((e) => `── ${e.event_type} ──`);
            return [...prev, ...lines, ...others].slice(-3000);
          });
        }
      } catch {
        /* server unreachable */
      }
      if (!stopped) setTimeout(pollEvents, 1000);
    };

    const pollStatus = async () => {
      try {
        const s = await api.getTrainingStatus();
        setStatus(s);
        if (s.running || (s.job && !s.job.ended_at)) {
          const m = await api.getTrainingMetrics();
          setMetrics(m.series);
        }
      } catch {
        /* ignore */
      }
      if (!stopped) setTimeout(pollStatus, 3000);
    };

    pollEvents();
    pollStatus();
    return () => {
      stopped = true;
    };
  }, []);

  // auto-scroll logs
  useEffect(() => {
    const el = logBoxRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs]);

  const loadHistory = useCallback(async () => {
    try {
      const d = await api.getTrainingHistory(50);
      setHistory(d.jobs);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    loadHistory();
    const t = window.setInterval(loadHistory, 10000);
    return () => window.clearInterval(t);
  }, [loadHistory]);

  // ---- actions ---------------------------------------------------------------
  const onPreflight = useCallback(async () => {
    setChecking(true);
    setIssues(null);
    try {
      const r = await api.trainingPreflight(formPayload());
      setIssues(r.issues);
      if (r.can_start) message.success("预检查通过，可以开始训练");
      else message.warning("预检查发现需要处理的问题");
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      message.error(`预检查失败: ${err.response?.data?.detail ?? err.message}`);
    } finally {
      setChecking(false);
    }
  }, [formPayload]);

  const onStart = useCallback(async () => {
    try {
      const s = await api.guidedStart(formPayload());
      setStatus(s);
      setLogs([]);
      setMetrics([]);
      seqRef.current = 0;
      message.success("训练已启动");
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      message.error(`启动失败: ${err.response?.data?.detail ?? err.message}`);
    }
  }, [formPayload]);

  const onStop = useCallback(async () => {
    await api.trainingStop();
    message.info("已请求停止");
  }, []);

  const onPrepareDataset = useCallback(async () => {
    setPreparing(true);
    try {
      const r = await api.prepareDataset({
        task_type: prepTask,
        dataset_ratio: prepRatio,
      });
      setData(r.data_yaml);
      setPrepOpen(false);
      const trainMatch = r.info.match(/Train images: (\d+)/);
      const valMatch = r.info.match(/Val images: (\d+)/);
      message.success(
        `训练集已生成：train ${trainMatch?.[1] ?? "?"} 张 / val ${valMatch?.[1] ?? "?"} 张，已填入 YAML 路径`,
        5
      );
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      message.error(`生成失败: ${err.response?.data?.detail ?? err.message}`);
    } finally {
      setPreparing(false);
    }
  }, [prepTask, prepRatio]);

  const metricGroups = [
    { key: "loss", title: "Loss" },
    { key: "quality", title: "Quality (mAP / P / R)" },
    { key: "learning_rate", title: "Learning Rate" },
  ];

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", background: "#f5f5f5" }}>
      <div
        style={{
          padding: "8px 16px",
          background: "#fff",
          borderBottom: "1px solid #f0f0f0",
          display: "flex",
          alignItems: "center",
          gap: 12,
        }}
      >
        <Button icon={<ArrowLeftOutlined />} onClick={onBack}>
          返回标注
        </Button>
        <span style={{ fontWeight: 600, fontSize: 16 }}>
          <ExperimentOutlined /> 训练中心
        </span>
        {status?.job && (
          <Tag color={running ? "processing" : "default"}>
            {status.job.display_name} · {status.job.status}
          </Tag>
        )}
      </div>

      <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
        {/* 左：新建任务 */}
        <div style={{ width: 380, overflow: "auto", padding: 12 }}>
          <Card size="small" title="新建训练任务（Ultralytics）" style={{ marginBottom: 12 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div>
                <div style={{ marginBottom: 4 }}>任务类型</div>
                <Select
                  style={{ width: "100%" }}
                  value={task}
                  onChange={setTask}
                  disabled={running}
                  options={["detect", "segment", "classify", "pose", "obb"].map((t) => ({
                    value: t,
                    label: t,
                  }))}
                />
              </div>
              <div>
                <div style={{ marginBottom: 4 }}>模型（.pt 路径或模型名）</div>
                <Input value={model} onChange={(e) => setModel(e.target.value)} disabled={running} />
              </div>
              <div>
                <div style={{ marginBottom: 4 }}>数据集 YAML</div>
                <Space.Compact style={{ width: "100%" }}>
                  <Input value={data} onChange={(e) => setData(e.target.value)} disabled={running} placeholder="data.yaml 路径" />
                  <Button icon={<FolderOpenOutlined />} onClick={() => setBrowse("data")} disabled={running} />
                </Space.Compact>
                <Button
                  type="link"
                  size="small"
                  style={{ padding: 0, marginTop: 4 }}
                  onClick={() => setPrepOpen(true)}
                  disabled={running}
                >
                  从当前标注数据集一键生成
                </Button>
              </div>
              <div>
                <div style={{ marginBottom: 4 }}>输出目录</div>
                <Space.Compact style={{ width: "100%" }}>
                  <Input value={project} onChange={(e) => setProject(e.target.value)} disabled={running} placeholder="runs 输出根目录" />
                  <Button icon={<FolderOpenOutlined />} onClick={() => setBrowse("project")} disabled={running} />
                </Space.Compact>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <div style={{ flex: 1 }}>
                  <div style={{ marginBottom: 4 }}>实验名</div>
                  <Input value={name} onChange={(e) => setName(e.target.value)} disabled={running} />
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ marginBottom: 4 }}>设备</div>
                  <Select
                    style={{ width: "100%" }}
                    value={device}
                    onChange={setDevice}
                    disabled={running}
                    options={[
                      { value: "cpu", label: "CPU" },
                      { value: "0", label: "GPU 0" },
                    ]}
                  />
                </div>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                {([
                  ["epochs", epochs, setEpochs],
                  ["batch", batch, setBatch],
                  ["imgsz", imgsz, setImgsz],
                ] as const).map(([label, value, setter]) => (
                  <div key={label} style={{ flex: 1 }}>
                    <div style={{ marginBottom: 4 }}>{label}</div>
                    <InputNumber
                      style={{ width: "100%" }}
                      min={1}
                      value={value}
                      onChange={(v) => setter(v ?? 1)}
                      disabled={running}
                    />
                  </div>
                ))}
              </div>
              <Collapse
                size="small"
                items={[
                  {
                    key: "adv",
                    label: "高级参数",
                    children: (
                      <div style={{ display: "flex", gap: 8 }}>
                        <div style={{ flex: 1 }}>
                          <div style={{ marginBottom: 4 }}>patience</div>
                          <InputNumber
                            style={{ width: "100%" }}
                            value={patience}
                            onChange={setPatience}
                            disabled={running}
                            placeholder="默认"
                          />
                        </div>
                        <div style={{ flex: 1 }}>
                          <div style={{ marginBottom: 4 }}>lr0</div>
                          <InputNumber
                            style={{ width: "100%" }}
                            step={0.001}
                            value={lr0}
                            onChange={setLr0}
                            disabled={running}
                            placeholder="默认"
                          />
                        </div>
                      </div>
                    ),
                  },
                ]}
              />
              <Space>
                <Button onClick={onPreflight} loading={checking} disabled={running || !data || !project}>
                  预检查
                </Button>
                {!running ? (
                  <Button type="primary" icon={<CaretRightOutlined />} onClick={onStart} disabled={!data || !project}>
                    开始训练
                  </Button>
                ) : (
                  <Button danger icon={<StopOutlined />} onClick={onStop}>
                    停止训练
                  </Button>
                )}
              </Space>
            </div>
          </Card>

          {issues && (
            <Card size="small" title={`预检查结果（${issues.filter((i) => i.severity === "error").length} 错误）`}>
              <List
                size="small"
                dataSource={issues}
                renderItem={(i) => (
                  <List.Item style={{ padding: "4px 0" }}>
                    <Tooltip title={i.suggestion}>
                      <span style={{ color: SEV_STYLE[i.severity]?.color, marginRight: 6 }}>
                        {SEV_STYLE[i.severity]?.icon}
                      </span>
                      {i.title}
                    </Tooltip>
                  </List.Item>
                )}
              />
            </Card>
          )}
        </div>

        {/* 右：日志 + 指标 + 历史 */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, padding: 12, gap: 12, overflow: "auto" }}>
          <Card size="small" title="实时日志" styles={{ body: { padding: 0 } }}>
            <div
              ref={logBoxRef}
              style={{
                height: 240,
                overflow: "auto",
                background: "#141414",
                color: "#d4d4d4",
                fontFamily: "Consolas, monospace",
                fontSize: 12,
                padding: 8,
                whiteSpace: "pre-wrap",
                wordBreak: "break-all",
              }}
            >
              {logs.length === 0 ? "（暂无日志）" : logs.join("\n")}
            </div>
          </Card>

          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            {metricGroups.map((g) => (
              <Card key={g.key} size="small" style={{ flex: "1 1 560px" }}>
                <LineChart
                  title={g.title}
                  series={metrics.filter((s) => s.group === g.key)}
                />
              </Card>
            ))}
          </div>

          <Card size="small" title="历史记录">
            <Table
              size="small"
              rowKey={(r) => String(r.job_id)}
              dataSource={history}
              pagination={{ pageSize: 8, size: "small" }}
              columns={[
                { title: "任务", dataIndex: "job_id", ellipsis: true },
                {
                  title: "状态",
                  dataIndex: "status",
                  width: 100,
                  render: (s: string) => (
                    <Tag color={s === "completed" ? "green" : s === "failed" ? "red" : "default"}>
                      {s}
                    </Tag>
                  ),
                },
                { title: "开始", dataIndex: "started_at", width: 170, ellipsis: true },
                { title: "best mAP50", dataIndex: "best_map50", width: 110 },
                { title: "final loss", dataIndex: "final_train_loss", width: 100 },
              ]}
            />
          </Card>
        </div>
      </div>

      <DirBrowserModal
        open={browse === "data"}
        title="选择数据集 YAML"
        fileExtensions={[".yaml", ".yml"]}
        onCancel={() => setBrowse(null)}
        onSelect={(p) => {
          setData(p);
          setBrowse(null);
        }}
      />
      <DirBrowserModal
        open={browse === "project"}
        title="选择输出目录"
        onCancel={() => setBrowse(null)}
        onSelect={(p) => {
          setProject(p);
          setBrowse(null);
        }}
      />

      <Modal
        open={prepOpen}
        title="从当前标注数据集生成训练集"
        okText="生成"
        cancelText="取消"
        confirmLoading={preparing}
        onCancel={() => setPrepOpen(false)}
        onOk={onPrepareDataset}
        width={420}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 14, marginTop: 8 }}>
          <div style={{ fontSize: 13, color: "#71717a" }}>
            将标注页当前打开的目录（Labelme JSON）转换为 YOLO 训练结构：
            自动提取类别、分层抽样划分 train/val、生成 data.yaml。
          </div>
          <div>
            <div style={{ marginBottom: 4 }}>任务类型</div>
            <Select
              style={{ width: "100%" }}
              value={prepTask}
              onChange={setPrepTask}
              options={["Detect", "OBB", "Segment", "Pose", "Classify"].map((t) => ({
                value: t,
                label: t,
              }))}
            />
          </div>
          <div>
            <div style={{ marginBottom: 4 }}>
              训练集比例：{(prepRatio * 100).toFixed(0)}%
            </div>
            <Slider
              min={0.5}
              max={0.95}
              step={0.05}
              value={prepRatio}
              onChange={setPrepRatio}
            />
          </div>
        </div>
      </Modal>
    </div>
  );
}
