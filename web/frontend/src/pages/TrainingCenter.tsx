import { useCallback, useEffect, useRef, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Collapse,
  Divider,
  Input,
  InputNumber,
  List,
  message,
  Modal,
  Radio,
  Select,
  Slider,
  Space,
  Splitter,
  Steps,
  Switch,
  Table,
  Tag,
  Tooltip,
  Typography,
} from "antd";
import {
  ArrowLeftOutlined,
  ApiOutlined,
  CaretRightOutlined,
  CheckCircleFilled,
  CloseCircleFilled,
  ExperimentOutlined,
  FolderOpenOutlined,
  QuestionCircleOutlined,
  StopOutlined,
  ThunderboltOutlined,
} from "@ant-design/icons";
import * as api from "../api/client";
import DatasetHealth from "../components/DatasetHealth";
import DirBrowserModal from "../components/DirBrowserModal";
import GuidedTour, { useGuidedTour, type GuidedTourStep } from "../components/GuidedTour";
import LineChart from "../components/LineChart";
import ModelPlayground from "../components/ModelPlayground";
import RemoteProfilesModal from "../components/RemoteProfilesModal";
import { useStudio } from "../store/useStudio";
import { readPanelWidth, savePanelWidth } from "../utils/panelStorage";

const TC_LEFT_WIDTH_KEY = "xaw_tc_left_width";

const SEV_STYLE: Record<string, { color: string; icon: React.ReactNode }> = {
  pass: { color: "#52c41a", icon: <CheckCircleFilled /> },
  warning: { color: "#faad14", icon: <CloseCircleFilled /> },
  error: { color: "#f5222d", icon: <CloseCircleFilled /> },
};

const MODELS_BY_TASK: Record<string, string[]> = {
  detect: ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "yolo11s.pt"],
  segment: ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolo11n-seg.pt"],
  obb: ["yolov8n-obb.pt", "yolov8s-obb.pt", "yolo11n-obb.pt"],
  classify: ["yolov8n-cls.pt", "yolov8s-cls.pt", "yolo11n-cls.pt"],
  pose: ["yolov8n-pose.pt", "yolo11n-pose.pt"],
};

const PREP_TASKS = ["Detect", "OBB", "Segment", "Classify", "Pose"];

const PARAM_TIPS: Record<string, string> = {
  epochs: "完整遍历数据集的次数；数据少用 30 即可快速验证",
  batch: "每批送入模型的图片数；-1 为自动，显存/内存不足可调小",
  imgsz: "训练输入图片边长；越大越精确，但更慢、更占资源",
  patience: "早停耐心：连续 N 轮指标无提升就提前停止；留空用默认",
  lr0: "初始学习率；一般留空用默认，太大容易不收敛",
};

const TERMINAL_STATUS = new Set(["completed", "failed", "stopped"]);

const TOUR_STEPS: GuidedTourStep[] = [
  {
    targetId: "tour-train-steps",
    title: "整体流程",
    description: "顶部步骤条展示当前进度：准备数据集 → 配置训练 → 训练中 → 产物与导出。",
  },
  {
    targetId: "tour-train-dataset",
    title: "数据集检查",
    description: "这里自动统计当前标注数据并推荐任务类型，开训前先确认数据没有问题。",
  },
  {
    targetId: "tour-train-exec",
    title: "执行位置",
    description: "默认在本机训练；有 GPU 服务器的话，也可以切换为远程训练。",
  },
  {
    targetId: "tour-train-params",
    title: "训练参数",
    description: "新手不用逐项调参，直接点「快速体验」等预设按钮即可一键填好。",
  },
  {
    targetId: "tour-train-start",
    title: "检查并启动",
    description: "点击后会先做一轮预检查，没有问题就自动开始训练。",
  },
  {
    targetId: "tour-train-logs",
    title: "日志与曲线",
    description: "训练过程中日志在这里实时滚动，下方还会展示 Loss、mAP 等曲线。",
  },
  {
    targetId: "tour-train-history",
    title: "历史记录与产物",
    description: "每次训练都会留档，点「产物」可以预览、下载或导出训练好的模型。",
  },
];

function formatEta(seconds: number): string {
  if (seconds < 60) return "不到 1 分钟";
  const m = Math.round(seconds / 60);
  if (m < 60) return `约 ${m} 分钟`;
  return `约 ${Math.floor(m / 60)} 小时 ${m % 60} 分钟`;
}

// ---- training-done notification (browser notification + beep) --------------
const NOTIFY_DONE_KEY = "xaw_notify_done";

const NOTIFY_STATUS_LABEL: Record<string, string> = {
  completed: "训练完成",
  failed: "训练失败",
  stopped: "训练已停止",
};

/** Ask for browser notification permission once, at a user-relevant moment. */
function requestNotifyPermission() {
  try {
    if (typeof Notification === "undefined") return;
    if (Notification.permission === "default") void Notification.requestPermission();
  } catch {
    /* notifications unsupported */
  }
}

function showDoneNotification(body: string) {
  try {
    if (typeof Notification === "undefined" || Notification.permission !== "granted") return;
    new Notification("训练已结束", { body });
  } catch {
    /* notifications unsupported */
  }
}

/** Short two-tone beep via Web Audio — no audio asset needed. */
function playDoneBeep() {
  try {
    const Ctor =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return;
    const ctx = new Ctor();
    const t0 = ctx.currentTime;
    [880, 1174.66].forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = freq;
      const start = t0 + i * 0.18;
      gain.gain.setValueAtTime(0.0001, start);
      gain.gain.exponentialRampToValueAtTime(0.15, start + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.16);
      osc.connect(gain).connect(ctx.destination);
      osc.start(start);
      osc.stop(start + 0.18);
    });
    window.setTimeout(() => void ctx.close(), 800);
  } catch {
    /* audio unavailable (autoplay policy etc.) */
  }
}

// ---- multi-run compare -------------------------------------------------------
// one color per job; dash pattern distinguishes metrics within the same job
const COMPARE_COLORS = [
  "#1677ff",
  "#52c41a",
  "#fa8c16",
  "#eb2f96",
  "#722ed1",
  "#13c2c2",
  "#f5222d",
  "#a0d911",
];
const COMPARE_DASHES = ["", "6 3", "2 2", "8 3 2 3"];

// ---- deploy wizard -----------------------------------------------------------
function joinOutputPath(outputDir: string, rel: string): string {
  const sep = outputDir.includes("\\") ? "\\" : "/";
  return outputDir.replace(/[/\\]+$/, "") + sep + rel.split("/").join(sep);
}

/** Ready-to-run inference sample with the real artifact path filled in. */
function buildDeploySnippet(absPath: string, isOnnx: boolean): string {
  if (isOnnx) {
    return [
      "# 先安装依赖: pip install onnxruntime numpy",
      "import numpy as np",
      "import onnxruntime as ort",
      "",
      `session = ort.InferenceSession(r"${absPath}", providers=["CPUExecutionProvider"])`,
      "inp = session.get_inputs()[0]",
      'print("模型输入:", inp.name, inp.shape)',
      "",
      "# 实际推理时把 dummy 换成你的图片预处理结果",
      "# (缩放到输入尺寸、归一化到 0~1、转成 NCHW 的 float32 数组)",
      "shape = [d if isinstance(d, int) else 1 for d in inp.shape]",
      "dummy = np.zeros(shape, dtype=np.float32)",
      "outputs = session.run(None, {inp.name: dummy})",
      'print("模型输出:", [o.shape for o in outputs])',
    ].join("\n");
  }
  return [
    "# 先安装依赖: pip install ultralytics",
    "from ultralytics import YOLO",
    "",
    `model = YOLO(r"${absPath}")  # 加载训练好的权重`,
    "",
    "# 对一张图片做推理（把路径换成你自己的图片）",
    'results = model.predict(source="your_image.jpg", save=True)',
    "results[0].show()  # 弹窗查看检测效果",
  ].join("\n");
}

interface Props {
  onBack: () => void;
}

export default function TrainingCenter({ onBack }: Props) {
  // form
  const [task, setTask] = useState("detect");
  const [model, setModel] = useState("yolov8n.pt");
  const [data, setData] = useState("");
  const [project, setProject] = useState("");
  const projectTouched = useRef(false);
  const [name, setName] = useState("train");
  const [device, setDevice] = useState("cpu");
  const [deviceInfo, setDeviceInfo] = useState<api.DeviceInfo | null>(null);
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
  const [quickstarting, setQuickstarting] = useState(false);
  const [starting, setStarting] = useState(false);
  const tour = useGuidedTour("xaw_tour_training_seen");
  const studioDir = useStudio((s) => s.dir);
  // 左栏宽度记忆：defaultSize 只在挂载时读一次，拖拽过程由 Splitter 内部维护（非受控）
  const [leftDefault] = useState(() => readPanelWidth(TC_LEFT_WIDTH_KEY, 380));

  // remote execution
  const [execMode, setExecMode] = useState<"local" | "remote_ssh">("local");
  const [remoteProfiles, setRemoteProfiles] = useState<api.RemoteProfile[]>([]);
  const [remoteProfileId, setRemoteProfileId] = useState<string | null>(null);
  const [remotePassword, setRemotePassword] = useState("");
  const [profilesOpen, setProfilesOpen] = useState(false);
  const [diagLoading, setDiagLoading] = useState(false);
  const [remoteDiag, setRemoteDiag] = useState<api.RemoteDiagnosticsResult | null>(null);
  const [remoteGpus, setRemoteGpus] = useState<api.RemoteGpuInfo[]>([]);
  const [deviceAutoHint, setDeviceAutoHint] = useState<string | null>(null);

  // runtime
  const [issues, setIssues] = useState<api.PreflightIssue[] | null>(null);
  const [datasetStats, setDatasetStats] = useState<api.DatasetStats | null>(null);
  const [datasetStatsError, setDatasetStatsError] = useState<string | null>(null);
  const [status, setStatus] = useState<api.TrainingStatusResponse | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<api.MetricSeries[]>([]);
  const [history, setHistory] = useState<Record<string, unknown>[]>([]);
  const [artifactJob, setArtifactJob] = useState<string | null>(null);
  const [artifactList, setArtifactList] = useState<api.ArtifactListResponse | null>(null);
  const [artifactLoading, setArtifactLoading] = useState(false);
  const [artifactError, setArtifactError] = useState<string | null>(null);
  const [artifactDetails, setArtifactDetails] = useState<api.ArtifactInfo[] | null>(null);
  const [artifactPreview, setArtifactPreview] = useState<{ path: string; failed: boolean } | null>(null);
  const [exportingPath, setExportingPath] = useState<string | null>(null);
  const [exportFormat, setExportFormat] = useState("onnx");
  const [registeringPath, setRegisteringPath] = useState<string | null>(null);
  // Playground 折叠面板受控展开(产物「试用」成功后自动展开)
  const [playgroundOpen, setPlaygroundOpen] = useState<string[]>([]);
  const [exportFormats, setExportFormats] = useState<api.TrainingExportFormat[] | null>(null);

  // 训练完成通知：默认开，localStorage 记忆；ref 供轮询闭包读最新值
  const [notifyDone, setNotifyDone] = useState(
    () => localStorage.getItem(NOTIFY_DONE_KEY) !== "0"
  );
  const notifyDoneRef = useRef(notifyDone);
  // 上一次 poll 到的 running，用于 true→false 跳变检测（只在页面开着期间发生
  // 的结束才通知，历史回放天然不会触发）
  const prevRunningRef = useRef(false);

  // 多 run 对比
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [compareOpen, setCompareOpen] = useState(false);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareData, setCompareData] = useState<
    { jobId: string; label: string; series: api.MetricSeries[] }[]
  >([]);

  // 部署向导
  const [deployArtifact, setDeployArtifact] = useState<api.ArtifactInfo | null>(null);

  const seqRef = useRef(0);
  const logBoxRef = useRef<HTMLDivElement>(null);
  // 轮询的 setTimeout 句柄，卸载时清理，避免组件销毁后仍发出一次请求
  const eventsTimerRef = useRef<number | undefined>(undefined);
  const statusTimerRef = useRef<number | undefined>(undefined);
  const running = !!status?.running;
  const selectedProfile =
    remoteProfiles.find((p) => p.profile_id === remoteProfileId) ?? null;
  // 密码认证的远程档案必须先填密码才能启动
  const remotePasswordMissing =
    execMode === "remote_ssh" &&
    selectedProfile?.auth_method === "password" &&
    remotePassword.trim().length === 0;

  // ---- wizard step derivation ------------------------------------------------
  // 步骤位置只由当前会话状态推导，不看历史记录（否则历史里一旦有跑完/失败的
  // 任务，步骤条会永远停在「产物与导出」）：
  // 训练中 → 2；当前任务到终态 → 3；表单(数据 YAML + 模型)就绪 → 2；
  // 仅有数据 → 1；否则 → 0
  const hasData = data.trim().length > 0;
  const formReady = hasData && model.trim().length > 0;
  const currentTerminal =
    !!status?.job && !running && TERMINAL_STATUS.has(String(status.job.status));
  const stepCurrent = running ? 2 : currentTerminal ? 3 : formReady ? 2 : hasData ? 1 : 0;
  const noDataset = datasetStatsError?.includes("No image directory") ?? false;

  const onTaskChange = useCallback(
    (t: string) => {
      setTask(t);
      const models = MODELS_BY_TASK[t] ?? [];
      if (!models.includes(model)) {
        setModel(models[0] ?? model);
      }
    },
    [model]
  );

  const applyPreset = useCallback((key: string) => {
    if (key === "quick") {
      setEpochs(30);
      setBatch(16);
      setImgsz(640);
      setPatience(null);
      setLr0(null);
    } else if (key === "standard") {
      setEpochs(100);
      setBatch(16);
      setImgsz(640);
      setPatience(null);
      setLr0(null);
    } else if (key === "high") {
      setEpochs(300);
      setPatience(50);
    }
  }, []);

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
      ...(execMode === "remote_ssh"
        ? {
            execution_mode: "remote_ssh",
            remote_profile_id: remoteProfileId,
            remote_password: remotePassword || null,
          }
        : { execution_mode: "local" }),
    }),
    [
      task, model, data, project, name, device, epochs, batch, imgsz,
      patience, lr0, execMode, remoteProfileId, remotePassword,
    ]
  );

  // ---- remote server profiles --------------------------------------------
  const loadRemoteProfiles = useCallback(async () => {
    try {
      const d = await api.listRemoteProfiles();
      setRemoteProfiles(d.profiles);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    loadRemoteProfiles();
  }, [loadRemoteProfiles]);

  // keep the selection valid as profiles are added/removed
  useEffect(() => {
    setRemoteProfileId((cur) =>
      cur && remoteProfiles.some((p) => p.profile_id === cur)
        ? cur
        : remoteProfiles[0]?.profile_id ?? null
    );
  }, [remoteProfiles]);

  const onDetectServer = useCallback(async () => {
    if (!remoteProfileId) {
      message.warning("请先选择或新建一个服务器档案");
      return;
    }
    setDiagLoading(true);
    setRemoteDiag(null);
    try {
      const d = await api.remoteDiagnostics(remoteProfileId, remotePassword || undefined);
      setRemoteDiag(d);
      setRemoteGpus(d.gpus);
      setDevice(d.recommended_device);
      setDeviceAutoHint(
        d.recommended_device === "cpu"
          ? "已按服务器自动选择 CPU"
          : `已按服务器自动选择 GPU ${d.recommended_device}`
      );
      message.success("服务器检测完成");
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      message.error(`服务器检测失败: ${err.response?.data?.detail ?? err.message}`, 6);
    } finally {
      setDiagLoading(false);
    }
  }, [remoteProfileId, remotePassword]);

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
      if (!stopped) eventsTimerRef.current = window.setTimeout(pollEvents, 1000);
    };

    const pollStatus = async () => {
      try {
        const s = await api.getTrainingStatus();
        setStatus(s);
        // running 跳变检测：基于每次 poll 到的最新值与 ref（轮询 effect 只挂载
        // 一次，直接读 state 会拿到闭包里的旧值）。
        // false→true：本页观察到训练开始，借此时机申请浏览器通知权限；
        // true→false 且 job 到终态：本次会话内的训练结束，触发通知。
        const wasRunning = prevRunningRef.current;
        prevRunningRef.current = !!s.running;
        if (!wasRunning && s.running && notifyDoneRef.current) {
          requestNotifyPermission();
        }
        const jobStatus = s.job ? String(s.job.status) : "";
        if (
          wasRunning &&
          !s.running &&
          s.job &&
          TERMINAL_STATUS.has(jobStatus) &&
          notifyDoneRef.current
        ) {
          const text = `${NOTIFY_STATUS_LABEL[jobStatus] ?? jobStatus} · ${
            s.job.display_name || s.job.job_id
          }`;
          if (jobStatus === "completed") message.success(text, 5);
          else if (jobStatus === "failed") message.error(text, 5);
          else message.info(text, 5);
          playDoneBeep();
          showDoneNotification(text);
        }
        if (s.running || (s.job && !s.job.ended_at)) {
          const m = await api.getTrainingMetrics();
          setMetrics(m.series);
        }
      } catch {
        /* ignore */
      }
      if (!stopped) statusTimerRef.current = window.setTimeout(pollStatus, 3000);
    };

    pollEvents();
    pollStatus();
    return () => {
      stopped = true;
      window.clearTimeout(eventsTimerRef.current);
      window.clearTimeout(statusTimerRef.current);
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

  const loadDatasetStats = useCallback(async () => {
    try {
      const s = await api.getDatasetStats();
      setDatasetStats(s);
      setDatasetStatsError(null);
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      setDatasetStats(null);
      setDatasetStatsError(err.response?.data?.detail ?? err.message);
    }
  }, []);

  // dataset stats on mount (the annotation page may already have a dir open)
  useEffect(() => {
    loadDatasetStats();
  }, [loadDatasetStats]);

  const loadArtifacts = useCallback(async (jobId: string) => {
    setArtifactLoading(true);
    setArtifactError(null);
    setArtifactPreview(null);
    try {
      const d = await api.getTrainingArtifacts(jobId);
      setArtifactList(d);
      setArtifactDetails(d.artifacts);
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      setArtifactError(err.response?.data?.detail ?? err.message);
      setArtifactList(null);
      setArtifactDetails(null);
    } finally {
      setArtifactLoading(false);
    }
  }, []);

  const onExportArtifact = useCallback(
    async (relPath: string) => {
      if (!artifactList) return;
      setExportingPath(relPath);
      const hide = message.loading(`正在导出为 ${exportFormat}…`, 0);
      try {
        const r = await api.exportModelArtifact(artifactList.job_id, relPath, exportFormat);
        message.success(`导出完成：${r.relative_path}`);
        await loadArtifacts(artifactList.job_id);
      } catch (e) {
        const err = e as { response?: { data?: { detail?: string } }; message: string };
        message.error(`导出失败: ${err.response?.data?.detail ?? err.message}`, 6);
      } finally {
        hide();
        setExportingPath(null);
      }
    },
    [artifactList, exportFormat, loadArtifacts]
  );

  const onTryModel = useCallback(
    async (relPath: string) => {
      if (!artifactList) return;
      setRegisteringPath(relPath);
      const hide = message.loading("正在注册并加载模型…", 0);
      try {
        const r = await api.registerModelArtifact(artifactList.job_id, relPath);
        await api.loadModel(r.config_file);
        // 试用 = 传图看效果:收起产物弹窗、展开 Playground 面板,
        // 用户直接拖图即可,不用再绕到标注页。
        setArtifactJob(null);
        setArtifactList(null);
        setArtifactError(null);
        setArtifactDetails(null);
        setArtifactPreview(null);
        setPlaygroundOpen(["playground"]);
        message.success("模型已加载,在「模型试用」里拖一张图片进去即可看到检测效果");
      } catch (e) {
        const err = e as { response?: { data?: { detail?: string } }; message: string };
        message.error(`试用失败: ${err.response?.data?.detail ?? err.message}`, 6);
      } finally {
        hide();
        setRegisteringPath(null);
      }
    },
    [artifactList]
  );

  // export format availability (env-checked on the backend)
  useEffect(() => {
    api
      .getTrainingExportFormats()
      .then((d) => {
        setExportFormats(d.formats);
        // 当前选中的格式在本机不可用时，自动切到第一个可用格式
        setExportFormat((cur) =>
          d.formats.some((f) => f.id === cur && f.available)
            ? cur
            : (d.formats.find((f) => f.available)?.id ?? cur)
        );
      })
      .catch(() => undefined);
  }, []);

  // static fallback keeps the dialog usable if the formats fetch fails
  const exportOptions = (
    exportFormats ?? [
      { id: "onnx", name: "ONNX", available: true, reason: null },
      { id: "engine", name: "TensorRT", available: true, reason: null },
      { id: "openvino", name: "OpenVINO", available: true, reason: null },
      { id: "tflite", name: "TFLite", available: true, reason: null },
      { id: "torchscript", name: "TorchScript", available: true, reason: null },
    ]
  ).map((f) => ({
    value: f.id,
    label: f.available ? f.name : `${f.name}（不可用）`,
    disabled: !f.available,
    title: f.reason ?? f.name,
  }));

  // device auto-detection on mount
  useEffect(() => {
    api
      .getDevice()
      .then((d) => {
        setDeviceInfo(d);
        setDevice(d.recommended);
      })
      .catch(() => undefined);
  }, []);

  // output dir follows the dataset location until the user overrides it
  useEffect(() => {
    if (projectTouched.current) return;
    const d = data.trim();
    if (!d) {
      setProject("");
      return;
    }
    const dir = d.replace(/[/\\][^/\\]*$/, "");
    if (dir && dir !== d) {
      setProject(`${dir}/runs`);
    }
  }, [data]);

  useEffect(() => {
    loadHistory();
    const t = window.setInterval(loadHistory, 10000);
    return () => window.clearInterval(t);
  }, [loadHistory]);

  // ---- actions ---------------------------------------------------------------
  const onQuickstart = useCallback(async () => {
    setQuickstarting(true);
    const hide = message.loading("正在自动生成数据集并启动训练…", 0);
    try {
      const r = await api.quickstart({ epochs });
      const trainMatch = r.dataset_info.match(/Train images: (\d+)/);
      const valMatch = r.dataset_info.match(/Val images: (\d+)/);
      message.success(
        `已自动开训：${r.task_type} · ${r.model} · ${r.device} · train ${trainMatch?.[1] ?? "?"} 张 / val ${valMatch?.[1] ?? "?"} 张`,
        6
      );
      void loadDatasetStats();
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      const detail = err.response?.data?.detail ?? err.message;
      if (String(detail).includes("No image directory")) {
        message.warning("请先在标注页打开一个数据集目录，再使用一键训练", 5);
      } else {
        message.error(`一键训练失败: ${detail}`, 5);
      }
    } finally {
      hide();
      setQuickstarting(false);
    }
  }, [epochs, loadDatasetStats]);

  const onStart = useCallback(async () => {
    if (execMode === "remote_ssh" && !remoteProfileId) {
      message.warning("请先选择或新建一个远程服务器档案");
      return;
    }
    setStarting(true);
    setIssues(null);
    try {
      let payload = formPayload();
      // 数据集 YAML 为空时的默认动作：标注页开着目录就从当前标注自动生成，
      // 免去小白先理解"什么是 data.yaml"这一步。
      if (!String(payload.data ?? "").trim()) {
        if (!studioDir) {
          message.warning(
            "还没有数据集：请先在标注页打开图片目录，或点「从当前标注数据集一键生成」"
          );
          return;
        }
        const prepTaskMap: Record<string, string> = {
          detect: "Detect", segment: "Segment", obb: "OBB",
          classify: "Classify", pose: "Pose",
        };
        const hide = message.loading(
          "数据集 YAML 为空，正在从当前标注目录自动生成训练集…", 0
        );
        try {
          const r = await api.prepareDataset({
            task_type: prepTaskMap[task] ?? "Detect",
            dataset_ratio: 0.9,
          });
          setData(r.data_yaml);
          payload = { ...payload, data: r.data_yaml };
          message.success("已自动生成训练集并填入，继续启动");
        } finally {
          hide();
        }
      }
      const pre = await api.trainingPreflight(payload);
      if (!pre.can_start) {
        setIssues(pre.issues);
        message.warning("预检查发现需要处理的问题，已阻止启动");
        return;
      }
      const s = await api.guidedStart(payload);
      setStatus(s);
      setLogs([]);
      setMetrics([]);
      seqRef.current = 0;
      message.success("训练已启动");
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      message.error(`启动失败: ${err.response?.data?.detail ?? err.message}`);
    } finally {
      setStarting(false);
    }
  }, [formPayload, execMode, remoteProfileId, studioDir, task]);

  const onStop = useCallback(async () => {
    await api.trainingStop();
    message.info("已请求停止");
  }, []);

  // ---- 完成通知开关 -------------------------------------------------------------
  const onNotifyDoneChange = useCallback((checked: boolean) => {
    setNotifyDone(checked);
    notifyDoneRef.current = checked;
    localStorage.setItem(NOTIFY_DONE_KEY, checked ? "1" : "0");
    if (checked) requestNotifyPermission();
  }, []);

  // ---- 多 run 对比 ---------------------------------------------------------------
  // 终态且有产物目录的任务才可选（没有输出目录的任务也拉不到 results.csv）
  const isComparable = useCallback(
    (r: Record<string, unknown>) =>
      TERMINAL_STATUS.has(String(r.status ?? "")) &&
      !!(r.output_directory || r.output_dir),
    []
  );

  const openCompare = useCallback(async () => {
    setCompareOpen(true);
    setCompareLoading(true);
    try {
      const jobs = await Promise.all(
        compareIds.map(async (id) => {
          const rec = history.find((h) => String(h.job_id) === id);
          const expName = String(rec?.display_name ?? "") || String(rec?.name ?? "") || id;
          // 实验名 + job 短 id（uuid 尾段），同名实验也能区分
          const label = `${expName} · ${id.slice(-6)}`;
          try {
            const d = await api.getTrainingHistoryMetrics(id);
            return { jobId: id, label, series: d.series };
          } catch {
            return { jobId: id, label, series: [] as api.MetricSeries[] };
          }
        })
      );
      setCompareData(jobs);
    } finally {
      setCompareLoading(false);
    }
  }, [compareIds, history]);

  // 同 group 的曲线叠加在一张图上：每个 job 一个颜色，同一 job 内的多个指标
  // 用虚线样式区分（图例文字为「实验名 · 指标名」）
  const compareGroupSeries = (group: string) =>
    compareData.flatMap((job, ji) =>
      job.series
        .filter((s) => s.group === group)
        .map((s, mi) => ({
          name: `${job.label} · ${s.name}`,
          points: s.points,
          color: COMPARE_COLORS[ji % COMPARE_COLORS.length],
          dash: COMPARE_DASHES[mi % COMPARE_DASHES.length] || undefined,
        }))
    );

  // ---- 部署向导 -------------------------------------------------------------------
  const deployInfo =
    deployArtifact && artifactList
      ? {
          isOnnx: deployArtifact.name.endsWith(".onnx"),
          absPath: joinOutputPath(artifactList.output_dir, deployArtifact.relative_path),
        }
      : null;
  const deploySnippet = deployInfo
    ? buildDeploySnippet(deployInfo.absPath, deployInfo.isOnnx)
    : "";

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
      void loadDatasetStats();
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      message.error(`生成失败: ${err.response?.data?.detail ?? err.message}`);
    } finally {
      setPreparing(false);
    }
  }, [prepTask, prepRatio, loadDatasetStats]);

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
        <Tooltip title="新手引导">
          <Button
            type="text"
            size="small"
            icon={<QuestionCircleOutlined />}
            onClick={tour.openTour}
          />
        </Tooltip>
        {status?.job && (
          <Tag color={running ? "processing" : "default"}>
            {status.job.display_name} · {status.job.status}
          </Tag>
        )}
      </div>

      <div
        id="tour-train-steps"
        style={{ padding: "6px 16px", background: "#fff", borderBottom: "1px solid #f0f0f0" }}
      >
        <Steps
          size="small"
          current={stepCurrent}
          items={[
            { title: "准备数据集" },
            { title: "配置训练" },
            { title: "训练中" },
            { title: "产物与导出" },
          ]}
        />
      </div>

      <Splitter
        style={{ flex: 1, minHeight: 0 }}
        onResize={(sizes) => savePanelWidth(TC_LEFT_WIDTH_KEY, sizes[0])}
      >
        {/* 左：新建任务 */}
        <Splitter.Panel defaultSize={leftDefault} min={300} max={640}>
        <div style={{ height: "100%", overflow: "auto", padding: 12 }}>
          {noDataset && (
            <Alert
              type="info"
              showIcon
              style={{ marginBottom: 12 }}
              message="请先在标注页打开图片目录，或直接使用一键训练"
            />
          )}
          <Card size="small" style={{ marginBottom: 12 }}>
            <Button
              type="primary"
              size="large"
              block
              icon={<ThunderboltOutlined />}
              loading={quickstarting}
              disabled={running || execMode === "remote_ssh"}
              onClick={onQuickstart}
            >
              一键训练（当前标注数据集）
            </Button>
            <div style={{ fontSize: 12, color: "#999", marginTop: 8 }}>
              {execMode === "remote_ssh"
                ? "远程服务器模式请使用下方表单配置训练。"
                : "自动推断任务类型、生成数据集、选择设备和模型，零配置直接开训。需要先在标注页打开数据集目录。"}
            </div>
          </Card>

          <Card id="tour-train-dataset" size="small" title="数据集检查" style={{ marginBottom: 12 }}>
            {datasetStats ? (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <div style={{ fontSize: 13 }}>
                  总图片数：<strong>{datasetStats.total_images}</strong>
                  <span style={{ color: "#999", marginLeft: 8 }}>
                    推荐任务：{datasetStats.recommended_task}
                  </span>
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                  {Object.entries(datasetStats.per_task_valid).map(([t, n]) => (
                    <Tag
                      key={t}
                      color={t === datasetStats.recommended_task ? "green" : "default"}
                    >
                      {t} {n}
                      {t === datasetStats.recommended_task ? " · 推荐" : ""}
                    </Tag>
                  ))}
                </div>
                {Object.keys(datasetStats.class_counts).length > 0 && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    <div style={{ fontSize: 12, color: "#999" }}>类别实例分布</div>
                    {Object.entries(datasetStats.class_counts)
                      .sort((a, b) => b[1] - a[1])
                      .map(([label, count]) => {
                        const max = Math.max(...Object.values(datasetStats.class_counts));
                        const pct = max > 0 ? (count / max) * 100 : 0;
                        return (
                          <div
                            key={label}
                            style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}
                          >
                            <span
                              style={{
                                width: 80,
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                                whiteSpace: "nowrap",
                                flexShrink: 0,
                              }}
                              title={label}
                            >
                              {label}
                            </span>
                            <div style={{ flex: 1, background: "#f0f0f0", borderRadius: 2, height: 12 }}>
                              <div
                                style={{
                                  width: `${pct}%`,
                                  height: "100%",
                                  background: "#1677ff",
                                  borderRadius: 2,
                                }}
                              />
                            </div>
                            <span style={{ width: 40, textAlign: "right", flexShrink: 0 }}>{count}</span>
                          </div>
                        );
                      })}
                  </div>
                )}
                {datasetStats.warnings.length > 0 && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    {datasetStats.warnings.map((w) => (
                      <div key={w.code} style={{ fontSize: 12, color: "#faad14" }}>
                        ⚠ {w.message}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div style={{ fontSize: 12, color: "#999" }}>
                {datasetStatsError
                  ? `暂无统计数据：${datasetStatsError}`
                  : "正在统计当前标注数据集…"}
              </div>
            )}
            <Divider style={{ margin: "12px 0" }} />
            <DatasetHealth dir={studioDir} onBack={onBack} />
          </Card>

          <Card id="tour-train-exec" size="small" title="执行位置" style={{ marginBottom: 12 }}>
            <Radio.Group
              size="small"
              optionType="button"
              buttonStyle="solid"
              value={execMode}
              onChange={(e) => {
                const mode = e.target.value as "local" | "remote_ssh";
                setExecMode(mode);
                // 切换执行位置后，另一侧的 GPU 列表 / 诊断结果 / 密码全部作废：
                // 本机回退到本地推荐设备，远程先回退 CPU，等重新检测
                setRemoteGpus([]);
                setRemoteDiag(null);
                setDeviceAutoHint(null);
                setRemotePassword("");
                setDevice(mode === "local" ? (deviceInfo?.recommended ?? "cpu") : "cpu");
              }}
              disabled={running}
              options={[
                { value: "local", label: "本机" },
                { value: "remote_ssh", label: "远程服务器" },
              ]}
            />
            {execMode === "remote_ssh" && (
              <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 10 }}>
                <div style={{ fontSize: 12, color: "#71717a", lineHeight: 1.7 }}>
                  数据集与权重在启动时从<b>本机自动上传</b>到服务器执行，训练结束后产物自动下载回本机的输出目录。
                  当前为全量上传，数据集大时请耐心等待日志里的上传阶段。
                </div>
                <div>
                  <div style={{ marginBottom: 4 }}>服务器档案</div>
                  <Space.Compact style={{ width: "100%" }}>
                    <Select
                      style={{ width: "100%" }}
                      value={remoteProfileId}
                      onChange={(v) => {
                        setRemoteProfileId(v);
                        // 换服务器档案后，上一台的 GPU 列表 / 诊断结果 / 密码作废，
                        // 设备回退 CPU，等重新检测
                        setRemoteGpus([]);
                        setRemoteDiag(null);
                        setDeviceAutoHint(null);
                        setRemotePassword("");
                        setDevice("cpu");
                      }}
                      disabled={running}
                      placeholder="选择服务器"
                      notFoundContent="暂无档案，请点「管理服务器」新增"
                      options={remoteProfiles.map((p) => ({
                        value: p.profile_id,
                        label: `${p.name}（${p.username}@${p.host}:${p.port}）`,
                        // 超长档案名下拉与悬停时可见全文
                        title: `${p.name}（${p.username}@${p.host}:${p.port}）`,
                      }))}
                    />
                    <Button onClick={() => setProfilesOpen(true)} disabled={running}>
                      管理服务器
                    </Button>
                  </Space.Compact>
                </div>
                {selectedProfile?.auth_method === "password" && (
                  <div>
                    <div style={{ marginBottom: 4 }}>登录密码（仅本次会话，不会保存）</div>
                    <Input.Password
                      value={remotePassword}
                      onChange={(e) => setRemotePassword(e.target.value)}
                      disabled={running}
                    />
                  </div>
                )}
                <div>
                  <Button
                    size="small"
                    icon={<ApiOutlined />}
                    loading={diagLoading}
                    disabled={!remoteProfileId || running}
                    onClick={onDetectServer}
                  >
                    检测服务器
                  </Button>
                  {deviceAutoHint && (
                    <span style={{ marginLeft: 8, fontSize: 12, color: "#52c41a" }}>
                      {deviceAutoHint}
                    </span>
                  )}
                </div>
                {remoteDiag && (
                  <div style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 2 }}>
                    {remoteDiag.items.map((it, idx) => (
                      <div
                        key={idx}
                        style={{
                          color:
                            it.status === "ERROR"
                              ? "#f5222d"
                              : it.status === "WARNING"
                                ? "#faad14"
                                : "#52c41a",
                          // 诊断消息可能含长 uname / 路径等无空格串，允许断行而不是溢出裁掉
                          whiteSpace: "normal",
                          wordBreak: "break-word",
                          overflowWrap: "anywhere",
                        }}
                      >
                        {it.status === "PASS" ? "✓" : it.status === "WARNING" ? "⚠" : "✗"}{" "}
                        {it.label}：{it.message.split("\n")[0]}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </Card>

          <Card id="tour-train-params" size="small" title="训练参数">
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div>
                <div style={{ marginBottom: 4 }}>任务类型</div>
                <Select
                  style={{ width: "100%" }}
                  value={task}
                  onChange={onTaskChange}
                  disabled={running}
                  options={["detect", "segment", "classify", "pose", "obb"].map((t) => ({
                    value: t,
                    label: t,
                  }))}
                />
              </div>
              <div>
                <div style={{ marginBottom: 4 }}>模型权重</div>
                <Select
                  style={{ width: "100%" }}
                  value={model}
                  onChange={setModel}
                  disabled={running}
                  options={(MODELS_BY_TASK[task] ?? [model]).map((m) => ({
                    value: m,
                    label: m,
                  }))}
                />
              </div>
              <div>
                <div style={{ marginBottom: 4 }}>数据集 YAML</div>
                <Space.Compact style={{ width: "100%" }}>
                  <Input value={data} onChange={(e) => setData(e.target.value)} disabled={running} placeholder="data.yaml 路径" title={data} />
                  <Button icon={<FolderOpenOutlined />} onClick={() => setBrowse("data")} disabled={running} />
                </Space.Compact>
                <Button
                  type="link"
                  size="small"
                  style={{ padding: 0, marginTop: 4 }}
                  onClick={() => {
                    const rec = datasetStats?.recommended_task;
                    if (rec && rec !== "Pose") setPrepTask(rec);
                    setPrepOpen(true);
                  }}
                  disabled={running}
                >
                  从当前标注数据集一键生成
                </Button>
              </div>
              <div>
                <div style={{ marginBottom: 4 }}>输出目录（默认跟随数据集位置，可修改）</div>
                <Space.Compact style={{ width: "100%" }}>
                  <Input
                    value={project}
                    onChange={(e) => {
                      projectTouched.current = true;
                      setProject(e.target.value);
                    }}
                    disabled={running}
                    placeholder="runs 输出根目录"
                    title={project}
                  />
                  <Button icon={<FolderOpenOutlined />} onClick={() => setBrowse("project")} disabled={running} />
                </Space.Compact>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <div style={{ flex: 1 }}>
                  <div style={{ marginBottom: 4 }}>实验名</div>
                  <Input value={name} onChange={(e) => setName(e.target.value)} disabled={running} />
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ marginBottom: 4 }}>
                    设备
                    <span style={{ fontSize: 11, color: "#999", marginLeft: 6 }}>
                      {execMode === "remote_ssh"
                        ? remoteGpus.length > 0
                          ? `服务器 ${remoteGpus[0].name}`
                          : "以服务器诊断为准"
                        : deviceInfo &&
                          (deviceInfo.gpus.length > 0 && deviceInfo.cuda_available
                            ? `检测到 ${deviceInfo.gpus[0].name}`
                            : "CPU 模式")}
                    </span>
                  </div>
                  <Select
                    style={{ width: "100%" }}
                    value={device}
                    onChange={setDevice}
                    disabled={running}
                    options={[
                      { value: "cpu", label: "CPU" },
                      ...(execMode === "remote_ssh"
                        ? remoteGpus.map((g) => ({
                            value: String(g.index),
                            label: `GPU ${g.index} · ${g.name}（远程）`,
                          }))
                        : deviceInfo?.cuda_available
                          ? deviceInfo.gpus.map((g) => ({
                              value: String(g.index),
                              label: `GPU ${g.index} · ${g.name}`,
                            }))
                          : []),
                    ]}
                  />
                </div>
              </div>
              <div>
                <div style={{ marginBottom: 4 }}>预设</div>
                <Radio.Group
                  size="small"
                  optionType="button"
                  buttonStyle="solid"
                  disabled={running}
                  // 受控且恒为空：预设是「动作」而非状态，value=null 保证同一
                  // 预设可重复点击应用，手动改参后也不会有误导性的高亮
                  value={null}
                  onChange={(e) => applyPreset(e.target.value as string)}
                  options={[
                    { value: "quick", label: "快速体验" },
                    { value: "standard", label: "标准训练" },
                    { value: "high", label: "高精度" },
                  ]}
                />
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                {([
                  ["epochs", epochs, setEpochs, 1, undefined],
                  ["batch", batch, setBatch, -1, undefined],
                  ["imgsz", imgsz, setImgsz, 32, 4096],
                ] as const).map(([label, value, setter, min, max]) => (
                  <div key={label} style={{ flex: 1 }}>
                    <div style={{ marginBottom: 4 }}>
                      <Tooltip title={PARAM_TIPS[label]}>{label}</Tooltip>
                    </div>
                    <InputNumber
                      style={{ width: "100%" }}
                      min={min}
                      max={max}
                      value={value}
                      // 0 不是合法取值：归一为该参数的最小值（batch 即 -1 自动）；
                      // 清空输入框静默回落最小值的行为保持不变
                      onChange={(v) => setter(v == null || v === 0 ? min : v)}
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
                          <div style={{ marginBottom: 4 }}>
                            <Tooltip title={PARAM_TIPS.patience}>patience</Tooltip>
                          </div>
                          <InputNumber
                            style={{ width: "100%" }}
                            min={0}
                            value={patience}
                            onChange={setPatience}
                            disabled={running}
                            placeholder="默认"
                          />
                        </div>
                        <div style={{ flex: 1 }}>
                          <div style={{ marginBottom: 4 }}>
                            <Tooltip title={PARAM_TIPS.lr0}>lr0</Tooltip>
                          </div>
                          <InputNumber
                            style={{ width: "100%" }}
                            min={0}
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
                {!running ? (
                  <Button
                    id="tour-train-start"
                    type="primary"
                    icon={<CaretRightOutlined />}
                    onClick={onStart}
                    loading={starting}
                    disabled={
                      !data ||
                      !project ||
                      (execMode === "remote_ssh" && !remoteProfileId) ||
                      remotePasswordMissing
                    }
                  >
                    检查并启动
                  </Button>
                ) : (
                  <Button danger icon={<StopOutlined />} onClick={onStop}>
                    停止训练
                  </Button>
                )}
                {remotePasswordMissing && !running && (
                  <span style={{ fontSize: 12, color: "#fa8c16" }}>
                    该服务器使用密码认证，请先填写密码
                  </span>
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
        </Splitter.Panel>

        {/* 右：模型试用 + 日志 + 指标 + 历史 */}
        <Splitter.Panel>
        <div style={{ height: "100%", display: "flex", flexDirection: "column", minWidth: 0, padding: 12, gap: 12, overflow: "auto" }}>
          <Collapse
            size="small"
            style={{ background: "#fff" }}
            activeKey={playgroundOpen}
            onChange={(keys) => setPlaygroundOpen(keys as string[])}
            items={[
              {
                key: "playground",
                label: (
                  <span style={{ fontWeight: 600 }}>
                    <ExperimentOutlined /> 模型试用（Playground）
                  </span>
                ),
                children: <ModelPlayground />,
              },
            ]}
          />
          <Card
            id="tour-train-logs"
            size="small"
            title="实时日志"
            styles={{ body: { padding: 0 } }}
            extra={
              <Space size={12}>
                <Tooltip title="训练结束（完成/失败/停止）时播放提示音并弹出浏览器通知">
                  <Space size={4}>
                    <Switch size="small" checked={notifyDone} onChange={onNotifyDoneChange} />
                    <span style={{ fontSize: 12, color: "#888", fontWeight: 400 }}>
                      完成时通知我
                    </span>
                  </Space>
                </Tooltip>
                {running && status?.eta_seconds != null && status.eta_seconds > 0 ? (
                  <span style={{ fontSize: 12, color: "#888", fontWeight: 400 }}>
                    预计剩余 {formatEta(status.eta_seconds)}
                  </span>
                ) : null}
              </Space>
            }
          >
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

          <Card
            id="tour-train-history"
            size="small"
            title="历史记录"
            extra={
              <Button
                size="small"
                disabled={compareIds.length < 2}
                onClick={openCompare}
              >
                对比选中{compareIds.length >= 2 ? `（${compareIds.length}）` : ""}
              </Button>
            }
          >
            <Table
              size="small"
              rowKey={(r) => String(r.job_id)}
              dataSource={history}
              pagination={{ pageSize: 8, size: "small" }}
              rowSelection={{
                selectedRowKeys: compareIds,
                onChange: (keys) => setCompareIds(keys.map(String)),
                getCheckboxProps: (r) => ({ disabled: !isComparable(r) }),
              }}
              columns={[
                { title: "ID", dataIndex: "job_id", ellipsis: true },
                {
                  title: "任务",
                  dataIndex: "task",
                  width: 80,
                  render: (t: string | null) => t || "-",
                },
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
                {
                  title: "产物",
                  render: (_: unknown, r: Record<string, unknown>) => (
                    <Button
                      size="small"
                      onClick={() => {
                        const id = String(r.job_id ?? "");
                        setArtifactJob(id);
                        void loadArtifacts(id);
                      }}
                    >
                      产物
                    </Button>
                  ),
                },
              ]}
            />
          </Card>
        </div>
        </Splitter.Panel>
      </Splitter>

      <Modal
        open={!!artifactJob}
        title="训练产物"
        onCancel={() => {
          setArtifactJob(null);
          setArtifactList(null);
          setArtifactError(null);
          setArtifactDetails(null);
          setArtifactPreview(null);
        }}
        footer={null}
        width={820}
      >
        {artifactLoading ? (
          <div>加载中…</div>
        ) : artifactError ? (
          <div style={{ color: "#f5222d" }}>{artifactError}</div>
        ) : artifactList ? (
          <>
            <div style={{ marginBottom: 8, color: "#666" }}>
              {artifactList.job_id} · {artifactList.output_dir}
            </div>
            {artifactDetails && artifactDetails.length > 0 ? (
              <List
                size="small"
                dataSource={artifactDetails}
                locale={{ emptyText: "暂无产物" }}
                renderItem={(a) => (
                  <List.Item
                    actions={[
                      ...(a.file_type === "png"
                        ? [
                            <Button
                              key="preview"
                              size="small"
                              onClick={() =>
                                setArtifactPreview((cur) =>
                                  cur?.path === a.relative_path
                                    ? null
                                    : { path: a.relative_path, failed: false }
                                )
                              }
                            >
                              {artifactPreview?.path === a.relative_path ? "收起" : "预览"}
                            </Button>,
                          ]
                        : []),
                      ...(a.name.endsWith(".pt")
                        ? [
                            <Space key="export" size={4}>
                              <Select
                                size="small"
                                value={exportFormat}
                                onChange={setExportFormat}
                                style={{ width: 130 }}
                                options={exportOptions}
                              />
                              <Button
                                size="small"
                                loading={exportingPath === a.relative_path}
                                disabled={exportingPath !== null}
                                onClick={() => onExportArtifact(a.relative_path)}
                              >
                                导出
                              </Button>
                            </Space>,
                          ]
                        : []),
                      ...(a.name.endsWith(".pt") || a.name.endsWith(".onnx")
                        ? [
                            <Button
                              key="try"
                              size="small"
                              type="primary"
                              ghost
                              loading={registeringPath === a.relative_path}
                              disabled={registeringPath !== null}
                              onClick={() => onTryModel(a.relative_path)}
                            >
                              试用
                            </Button>,
                            <Button
                              key="deploy"
                              size="small"
                              onClick={() => setDeployArtifact(a)}
                            >
                              部署
                            </Button>,
                          ]
                        : []),
                      <Button
                        key="download"
                        size="small"
                        href={api.trainingArtifactDownloadUrl(artifactList.job_id, a.relative_path)}
                        target="_blank"
                      >
                        下载
                      </Button>,
                    ]}
                  >
                    <div style={{ width: "100%" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                        <strong>{a.name}</strong>
                        <span style={{ color: "#999" }}>{a.file_type}</span>
                      </div>
                      <div style={{ fontSize: 12, color: "#999", wordBreak: "break-all" }}>{a.relative_path}</div>
                      <div style={{ fontSize: 12, color: "#999" }}>
                        {a.size} bytes · {a.modified_at}
                      </div>
                      {artifactPreview?.path === a.relative_path &&
                        (artifactPreview.failed ? (
                          <div style={{ fontSize: 12, color: "#999", marginTop: 8 }}>
                            图片加载失败
                          </div>
                        ) : (
                          <img
                            src={api.trainingArtifactDownloadUrl(artifactList.job_id, a.relative_path)}
                            alt={a.name}
                            style={{
                              maxWidth: "100%",
                              marginTop: 8,
                              border: "1px solid #f0f0f0",
                              borderRadius: 4,
                            }}
                            onError={() =>
                              setArtifactPreview((cur) =>
                                cur ? { ...cur, failed: true } : cur
                              )
                            }
                          />
                        ))}
                    </div>
                  </List.Item>
                )}
              />
            ) : (
              <div>暂无产物</div>
            )}
            <div style={{ marginTop: 12 }}>
              <Button href={api.trainingArtifactDownloadAllUrl(artifactList.job_id)} target="_blank">
                下载全部 ZIP
              </Button>
            </div>
          </>
        ) : (
          <div>暂无产物</div>
        )}
      </Modal>

      <Modal
        open={compareOpen}
        title={`训练对比（${compareData.length || compareIds.length} 个任务）`}
        onCancel={() => setCompareOpen(false)}
        footer={null}
        width={920}
      >
        {compareLoading ? (
          <div>加载中…</div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {metricGroups.map((g) => (
              <LineChart
                key={g.key}
                title={g.title}
                series={compareGroupSeries(g.key)}
                width={860}
                height={240}
              />
            ))}
            {compareData.every((j) => j.series.length === 0) && (
              <div style={{ fontSize: 12, color: "#999" }}>
                所选任务都没有可对比的指标：任务需跑到至少第 1 个 epoch 并生成 results.csv
                （远程任务的产物在结束时回传到本机后才能对比）。
              </div>
            )}
          </div>
        )}
      </Modal>

      <Modal
        open={!!deployArtifact}
        title={deployArtifact ? `部署：${deployArtifact.name}` : "部署"}
        onCancel={() => setDeployArtifact(null)}
        footer={null}
        width={700}
      >
        {deployInfo && (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div style={{ fontSize: 13, color: "#555", lineHeight: 1.8 }}>
              {deployInfo.isOnnx
                ? "这是 ONNX 模型文件：跨平台、不依赖 PyTorch，装个 onnxruntime 就能跑推理。"
                : "这是 PyTorch 权重文件（.pt）：ultralytics 官方格式，直接推理或继续训练都可以。"}
              远程服务器训练的产物在任务结束时已自动回传到本机，下面的路径就是本机文件，可直接使用。
            </div>
            <div>
              <div style={{ marginBottom: 4, fontWeight: 600 }}>文件路径</div>
              <Typography.Text
                copyable
                code
                style={{ wordBreak: "break-all", whiteSpace: "normal" }}
              >
                {deployInfo.absPath}
              </Typography.Text>
            </div>
            <div>
              <div style={{ marginBottom: 4, fontWeight: 600 }}>
                示例推理代码（先安装依赖：
                {deployInfo.isOnnx ? "pip install onnxruntime numpy" : "pip install ultralytics"}）
              </div>
              <div style={{ position: "relative" }}>
                <pre
                  style={{
                    background: "#141414",
                    color: "#d4d4d4",
                    padding: 12,
                    borderRadius: 6,
                    fontSize: 12,
                    lineHeight: 1.6,
                    overflow: "auto",
                    margin: 0,
                  }}
                >
                  {deploySnippet}
                </pre>
                <Typography.Text
                  copyable={{ text: deploySnippet, tooltips: ["复制代码", "已复制"] }}
                  style={{ position: "absolute", top: 6, right: 8 }}
                />
              </div>
            </div>
            <div style={{ fontSize: 12, color: "#999", lineHeight: 1.8 }}>
              代码里的路径已填好真实路径，复制后可直接运行；想放到其他机器上用，
              把文件拷过去并改一下代码里的路径即可。
            </div>
          </div>
        )}
      </Modal>

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
          projectTouched.current = true;
          setProject(p);
          setBrowse(null);
        }}
      />

      <RemoteProfilesModal
        open={profilesOpen}
        onClose={() => setProfilesOpen(false)}
        onProfilesChanged={setRemoteProfiles}
      />

      <GuidedTour steps={TOUR_STEPS} open={tour.open} onClose={tour.closeTour} />

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
              options={PREP_TASKS.map((t) => ({
                value: t,
                label: t === "Pose" ? "Pose（Web 端暂不支持）" : t,
                disabled: t === "Pose",
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
