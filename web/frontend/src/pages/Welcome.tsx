import { useRef, useState } from "react";
import { message, Modal } from "antd";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import {
  FolderOpenOutlined,
  ArrowRightOutlined,
  ThunderboltOutlined,
  VideoCameraOutlined,
  ExperimentOutlined,
  CloudServerOutlined,
  DisconnectOutlined,
} from "@ant-design/icons";
import DirBrowserModal from "../components/DirBrowserModal";
import { useStudio } from "../store/useStudio";
import { getServerUrl, getToken, setServerUrl, setToken } from "../api/client";
import "@fontsource-variable/outfit";
import "./welcome.css";

gsap.registerPlugin(useGSAP);

const FEATURES = [
  { icon: <ThunderboltOutlined />, title: "AI 自动标注", desc: "190+ 模型一键预标注" },
  { icon: <VideoCameraOutlined />, title: "视频跟踪", desc: "MOT 跨帧目标追踪" },
  { icon: <ExperimentOutlined />, title: "训练中心", desc: "Ultralytics 一站式训练" },
];

export default function Welcome() {
  const openDir = useStudio((s) => s.openDir);
  const openVideo = useStudio((s) => s.openVideo);
  const [path, setPath] = useState("");
  const [browserOpen, setBrowserOpen] = useState(false);
  const [opening, setOpening] = useState(false);
  const [remoteOpen, setRemoteOpen] = useState(false);
  const [serverInput, setServerInput] = useState(getServerUrl());
  const [tokenInput, setTokenInput] = useState(getToken());
  const scope = useRef<HTMLDivElement>(null);
  const serverUrl = getServerUrl();

  // Auto-resume disabled — always show Welcome page on startup.
  // To re-enable, uncomment the useEffect below.
  // const resumedRef = useRef(false);
  // useEffect(() => {
  //   if (resumedRef.current) return;
  //   resumedRef.current = true;
  //   try {
  //     const raw = localStorage.getItem("xaw_last_session");
  //     if (!raw) return;
  //     const s = JSON.parse(raw) as { type?: string; path?: string };
  //     if (!s.path) return;
  //     const hide = message.loading(`恢复上次会话：${s.path}`, 0);
  //     const p = s.type === "video" ? openVideo(s.path) : openDir(s.path);
  //     p.catch(() => undefined).finally(hide);
  //   } catch {
  //     /* no session to resume */
  //   }
  //   // eslint-disable-next-line react-hooks/exhaustive-deps
  // }, []);

  useGSAP(
    () => {
      const mm = gsap.matchMedia();
      mm.add("(prefers-reduced-motion: no-preference)", () => {
        const tl = gsap.timeline({ defaults: { ease: "power3.out" } });
        tl.from(".wl-brand", { y: 16, opacity: 0, duration: 0.5 })
          .from(".wl-headline .line", { y: 34, opacity: 0, stagger: 0.09, duration: 0.65 }, "-=0.25")
          .from(".wl-sub", { y: 18, opacity: 0, duration: 0.5 }, "-=0.3")
          .from(".wl-panel", { y: 22, opacity: 0, duration: 0.55 }, "-=0.25")
          .from(".wl-feature", { y: 16, opacity: 0, stagger: 0.08, duration: 0.45 }, "-=0.25")
          .from(".wl-visual", { x: 60, opacity: 0, duration: 0.8, ease: "power2.out" }, "-=0.6")
          .from(".wl-foot", { opacity: 0, duration: 0.5 }, "-=0.3");
      });
      // reduced-motion: everything stays visible, no animation
    },
    { scope }
  );

  const doOpen = async (p: string) => {
    if (!p.trim() || opening) return;
    setOpening(true);
    try {
      await openDir(p.trim());
    } catch (e) {
      message.error(`打开失败: ${(e as Error).message}`);
      setOpening(false);
    }
  };

  return (
    <div className="wl-root" ref={scope}>
      <div className="wl-layout">
        {/* left: lead column */}
        <div className="wl-left">
          <div className="wl-brand">
            <span className="wl-brand-mark" aria-hidden>
              <svg viewBox="0 0 48 48" width="26" height="26" fill="none">
                <rect x="6" y="10" width="22" height="16" rx="2.5" stroke="#2563eb" strokeWidth="3" />
                <path d="M32 18l8 5-8 5v-10z" fill="#2563eb" />
                <circle cx="14" cy="35" r="4" stroke="#18181b" strokeWidth="3" />
                <path d="M24 35h12" stroke="#18181b" strokeWidth="3" strokeLinecap="round" />
              </svg>
            </span>
            <span className="wl-brand-name">TrainLens</span>
          </div>

          <h1 className="wl-headline">
            <span className="line">AI 加速的</span>
            <span className="line">图像标注工作台</span>
          </h1>

          <p className="wl-sub">
            从手动标注、AI 预标注到视频跟踪与模型训练，一个页面完成。标注 JSON 与桌面版完全兼容。
          </p>

          <div className="wl-panel">
            <label className="wl-label" htmlFor="wl-dir-input">
              图片目录
            </label>
            <div className="wl-input-row">
              <input
                id="wl-dir-input"
                className="wl-input"
                placeholder="输入路径，或点击右侧浏览"
                value={path}
                onChange={(e) => setPath(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && doOpen(path)}
              />
              <button type="button" className="wl-btn-ghost" onClick={() => setBrowserOpen(true)}>
                <FolderOpenOutlined /> 浏览
              </button>
              <button
                type="button"
                className="wl-btn-primary"
                onClick={() => doOpen(path)}
                disabled={!path.trim() || opening}
              >
                {opening ? "打开中" : "打开目录"}
                {!opening && <ArrowRightOutlined />}
              </button>
            </div>
            <button type="button" className="wl-demo" onClick={() => doOpen("D:/x-anylabeling/assets")}>
              使用示例目录快速体验
            </button>
          </div>

          <button
            type="button"
            className="wl-remote"
            onClick={() => {
              setServerInput(getServerUrl());
              setTokenInput(getToken());
              setRemoteOpen(true);
            }}
          >
            <CloudServerOutlined />
            {serverUrl ? `已连接：${serverUrl}` : "连接远程服务器"}
          </button>

          <div className="wl-features">
            {FEATURES.map((f) => (
              <div key={f.title} className="wl-feature">
                <span className="wl-feature-icon">{f.icon}</span>
                <div>
                  <div className="wl-feature-title">{f.title}</div>
                  <div className="wl-feature-desc">{f.desc}</div>
                </div>
              </div>
            ))}
          </div>

          <div className="wl-foot">A / D 切换图片，Ctrl+S 保存，R 矩形，P 多边形，V 选择</div>
        </div>

        {/* right: real product visual, bleeding off the right edge */}
        <div className="wl-right">
          <figure className="wl-visual">
            <img src="/annotation-preview.jpg" alt="TrainLens 真实标注效果预览" />
            <figcaption>真实标注效果：assets/demo.jpg</figcaption>
          </figure>
        </div>
      </div>

      <DirBrowserModal
        open={browserOpen}
        onCancel={() => setBrowserOpen(false)}
        onSelect={async (p) => {
          setBrowserOpen(false);
          await doOpen(p);
        }}
      />

      <Modal
        open={remoteOpen}
        title="连接远程服务器"
        okText="保存并连接"
        cancelText="取消"
        onCancel={() => setRemoteOpen(false)}
        onOk={() => {
          setServerUrl(serverInput);
          setToken(tokenInput.trim());
          setRemoteOpen(false);
          if (serverInput.trim()) {
            message.success(`已连接到 ${serverInput.trim()}`);
          }
          window.location.reload();
        }}
        width={420}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 8 }}>
          <div>
            <div style={{ marginBottom: 4 }}>服务器地址</div>
            <input
              className="wl-input"
              style={{ width: "100%" }}
              placeholder="例如 http://172.20.10.6:8000"
              value={serverInput}
              onChange={(e) => setServerInput(e.target.value)}
            />
          </div>
          <div>
            <div style={{ marginBottom: 4 }}>访问令牌（服务器启动时控制台打印）</div>
            <input
              className="wl-input"
              style={{ width: "100%" }}
              type="password"
              placeholder="无令牌可留空"
              value={tokenInput}
              onChange={(e) => setTokenInput(e.target.value)}
            />
          </div>
          <div style={{ fontSize: 12, color: "#71717a" }}>
            留空地址则使用本机后端。连接后，标注、AI 推理、训练全部在远程服务器上执行。
          </div>
          {serverUrl && (
            <button
              type="button"
              className="wl-demo"
              style={{ alignSelf: "flex-start", color: "#dc2626" }}
              onClick={() => {
                setServerUrl("");
                setToken("");
                setRemoteOpen(false);
                message.success("已断开远程连接，使用本机后端");
                window.location.reload();
              }}
            >
              <DisconnectOutlined /> 断开当前连接（{serverUrl}）
            </button>
          )}
        </div>
      </Modal>
    </div>
  );
}
