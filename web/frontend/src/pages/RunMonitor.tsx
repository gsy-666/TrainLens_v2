import { useCallback, useEffect, useRef, useState } from "react";
import {
  Button,
  Card,
  Input,
  List,
  message,
  Select,
  Space,
  Tag,
} from "antd";
import {
  ArrowLeftOutlined,
  CaretRightOutlined,
  FolderOpenOutlined,
  MonitorOutlined,
  ReloadOutlined,
  StopOutlined,
} from "@ant-design/icons";
import * as api from "../api/client";
import DirBrowserModal from "../components/DirBrowserModal";
import LineChart from "../components/LineChart";

interface Props {
  onBack: () => void;
}

export default function RunMonitor({ onBack }: Props) {
  const [workspace, setWorkspace] = useState("");
  const [browseOpen, setBrowseOpen] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [wsInfo, setWsInfo] = useState<api.WorkspaceInfo | null>(null);
  const [script, setScript] = useState("");
  const [pythonPath, setPythonPath] = useState("");
  const [args, setArgs] = useState("");
  const [running, setRunning] = useState(false);
  const [run, setRun] = useState<api.RunInfo | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [samples, setSamples] = useState<api.ResourceSample[]>([]);

  const seqRef = useRef(0);
  const logBoxRef = useRef<HTMLDivElement>(null);

  // ---- polling ---------------------------------------------------------------
  useEffect(() => {
    let stopped = false;
    const poll = async () => {
      try {
        const d = await api.monitorLogs(seqRef.current);
        seqRef.current = d.latest;
        if (d.lines.length > 0) {
          setLogs((prev) =>
            [...prev, ...d.lines.map((l) => l.line)].slice(-3000)
          );
        }
        const s = await api.monitorStatus();
        setRunning(s.running);
        setRun(s.run);
        const res = await api.monitorResources(300);
        setSamples(res.samples);
      } catch {
        /* ignore */
      }
      if (!stopped) setTimeout(poll, 1000);
    };
    poll();
    return () => {
      stopped = true;
    };
  }, []);

  useEffect(() => {
    const el = logBoxRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs]);

  // ---- actions -----------------------------------------------------------------
  const onScan = useCallback(async () => {
    if (!workspace.trim()) {
      message.warning("请先选择工作目录");
      return;
    }
    setScanning(true);
    try {
      const info = await api.monitorScan(workspace.trim());
      setWsInfo(info);
      if (info.detected_scripts.length > 0) setScript(info.detected_scripts[0].path);
      if (info.detected_environments.length > 0)
        setPythonPath(info.detected_environments[0].python_path);
      message.success(
        `扫描完成：${info.detected_scripts.length} 个脚本，${info.detected_environments.length} 个环境`
      );
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      message.error(`扫描失败: ${err.response?.data?.detail ?? err.message}`);
    } finally {
      setScanning(false);
    }
  }, [workspace]);

  const onStart = useCallback(async () => {
    if (!script) {
      message.warning("请选择脚本");
      return;
    }
    try {
      const r = await api.monitorStart({
        workspace,
        script_path: script,
        python_path: pythonPath,
        arguments: args,
      });
      setRun(r);
      setRunning(true);
      message.success(`已启动 pid=${r.pid}`);
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      message.error(`启动失败: ${err.response?.data?.detail ?? err.message}`);
    }
  }, [workspace, script, pythonPath, args]);

  const onStop = useCallback(async () => {
    const r = await api.monitorStop();
    if (r.stopped) message.info("已停止");
    else message.warning(`停止失败: ${r.reason ?? "未知原因"}`);
  }, []);

  // ---- chart data ----------------------------------------------------------------
  const t0 = samples.length > 0 ? samples[0].ts : 0;
  const toSeries = (name: string, pick: (s: api.ResourceSample) => number | undefined) => ({
    name,
    group: "res",
    points: samples
      .filter((s) => pick(s) !== undefined)
      .map((s) => [Math.round(s.ts - t0), pick(s)!] as [number, number]),
  });
  const cpuSeries = [
    toSeries("进程 CPU%", (s) => s.proc_cpu),
    toSeries("系统 CPU%", (s) => s.system_cpu),
  ].filter((s) => s.points.length > 0);
  const memSeries = [
    toSeries("进程 RSS MB", (s) => s.proc_rss_mb),
    toSeries("系统内存%", (s) => s.system_mem_percent),
  ].filter((s) => s.points.length > 0);
  const gpuSeries = [
    toSeries("GPU 利用率%", (s) => s.gpu_util),
    toSeries("GPU 显存 MB", (s) => s.gpu_mem_used_mb),
  ].filter((s) => s.points.length > 0);

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
          <MonitorOutlined /> 运行监控
        </span>
        {run && (
          <Tag color={running ? "processing" : run.status === "completed" ? "success" : "default"}>
            {run.script_path.split(/[/\\]/).pop()} · {running ? "running" : run.status}
            {run.exit_code !== null ? ` (${run.exit_code})` : ""}
          </Tag>
        )}
      </div>

      <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
        {/* 左：配置 */}
        <div style={{ width: 380, overflow: "auto", padding: 12 }}>
          <Card size="small" title="工作区" style={{ marginBottom: 12 }}>
            <Space.Compact style={{ width: "100%" }}>
              <Input
                value={workspace}
                onChange={(e) => setWorkspace(e.target.value)}
                placeholder="训练项目目录"
              />
              <Button icon={<FolderOpenOutlined />} onClick={() => setBrowseOpen(true)} />
              <Button icon={<ReloadOutlined />} onClick={onScan} loading={scanning}>
                扫描
              </Button>
            </Space.Compact>
            {wsInfo && (
              <List
                size="small"
                style={{ marginTop: 8 }}
                header={<b>检测到的脚本</b>}
                dataSource={wsInfo.detected_scripts}
                locale={{ emptyText: "未检测到训练脚本" }}
                renderItem={(s) => (
                  <List.Item
                    onClick={() => setScript(s.path)}
                    style={{
                      cursor: "pointer",
                      padding: "4px 8px",
                      background: script === s.path ? "#e6f4ff" : undefined,
                    }}
                  >
                    <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis" }}>
                      {s.path.split(/[/\\]/).pop()}
                    </span>
                    {s.framework && <Tag>{s.framework}</Tag>}
                  </List.Item>
                )}
              />
            )}
          </Card>

          <Card size="small" title="启动配置">
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div>
                <div style={{ marginBottom: 4 }}>脚本路径</div>
                <Input value={script} onChange={(e) => setScript(e.target.value)} disabled={running} />
              </div>
              <div>
                <div style={{ marginBottom: 4 }}>Python 解释器</div>
                <Select
                  style={{ width: "100%" }}
                  value={pythonPath || undefined}
                  onChange={setPythonPath}
                  disabled={running}
                  placeholder="默认：当前后端 Python"
                  allowClear
                  options={(wsInfo?.detected_environments ?? []).map((e) => ({
                    value: e.python_path,
                    label: `${e.env_type} · ${e.python_path}`,
                  }))}
                />
                <Input
                  style={{ marginTop: 4 }}
                  value={pythonPath}
                  onChange={(e) => setPythonPath(e.target.value)}
                  disabled={running}
                  placeholder="或手动输入解释器路径"
                />
              </div>
              <div>
                <div style={{ marginBottom: 4 }}>命令行参数</div>
                <Input
                  value={args}
                  onChange={(e) => setArgs(e.target.value)}
                  disabled={running}
                  placeholder="例如 --epochs 100 --batch 8"
                />
              </div>
              <Space>
                {!running ? (
                  <Button type="primary" icon={<CaretRightOutlined />} onClick={onStart} disabled={!script}>
                    启动
                  </Button>
                ) : (
                  <Button danger icon={<StopOutlined />} onClick={onStop}>
                    停止
                  </Button>
                )}
              </Space>
            </div>
          </Card>
        </div>

        {/* 右：日志 + 资源 */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, padding: 12, gap: 12, overflow: "auto" }}>
          <Card size="small" title="控制台输出" styles={{ body: { padding: 0 } }}>
            <div
              ref={logBoxRef}
              style={{
                height: 260,
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
              {logs.length === 0 ? "（暂无输出）" : logs.join("\n")}
            </div>
          </Card>

          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <Card size="small" style={{ flex: "1 1 420px" }}>
              <LineChart title="CPU" series={cpuSeries} width={420} />
            </Card>
            <Card size="small" style={{ flex: "1 1 420px" }}>
              <LineChart title="内存" series={memSeries} width={420} />
            </Card>
            {gpuSeries.length > 0 && (
              <Card size="small" style={{ flex: "1 1 420px" }}>
                <LineChart title="GPU" series={gpuSeries} width={420} />
              </Card>
            )}
          </div>
        </div>
      </div>

      <DirBrowserModal
        open={browseOpen}
        title="选择工作目录"
        onCancel={() => setBrowseOpen(false)}
        onSelect={(p) => {
          setWorkspace(p);
          setBrowseOpen(false);
        }}
      />
    </div>
  );
}
