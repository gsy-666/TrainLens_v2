import { Button, Space } from "antd";
import { ExperimentOutlined, MonitorOutlined } from "@ant-design/icons";
import { useEffect, useRef, useState } from "react";
import LabelStudio from "./pages/LabelStudio";
import TrainingCenter from "./pages/TrainingCenter";
import RunMonitor from "./pages/RunMonitor";
import Welcome from "./pages/Welcome";
import TokenGate from "./pages/TokenGate";
import { useStudio } from "./store/useStudio";

type View = "label" | "training" | "monitor";

export default function App() {
  const dir = useStudio((s) => s.dir);
  const video = useStudio((s) => s.video);
  const [view, setView] = useState<View>("label");
  const [needsAuth, setNeedsAuth] = useState(false);
  const inSession = Boolean(dir || video);

  useEffect(() => {
    const onUnauthorized = () => setNeedsAuth(true);
    window.addEventListener("xaw:unauthorized", onUnauthorized);
    return () => window.removeEventListener("xaw:unauthorized", onUnauthorized);
  }, []);

  // ---- browser history wiring -------------------------------------------------
  // The SPA switches pages with internal state only; without pushing history
  // entries, the browser Back button would leave the app entirely. Mirror every
  // navigation into window.history so Back returns to the previous page
  // (label studio -> training center -> ... -> welcome).
  const navRef = useRef<string | null>(null);
  useEffect(() => {
    const key = inSession ? view : "welcome";
    if (navRef.current === null) {
      // mark the entry the app landed on
      window.history.replaceState({ view: key }, "");
      navRef.current = key;
      return;
    }
    if (navRef.current !== key) {
      navRef.current = key;
      window.history.pushState({ view: key }, "");
    }
  }, [view, inSession]);

  useEffect(() => {
    const onPop = (e: PopStateEvent) => {
      const v = (e.state as { view?: string } | null)?.view;
      const st = useStudio.getState();
      const hasSession = Boolean(st.dir || st.video);
      if ((v === "training" || v === "monitor" || v === "label") && hasSession) {
        setView(v);
        navRef.current = v;
      } else {
        // popped past the app root (or an entry that needs a session): Welcome
        st.closeSession();
        setView("label");
        navRef.current = "welcome";
      }
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  if (needsAuth) {
    return <TokenGate />;
  }

  if (!dir && !video) {
    return <Welcome />;
  }

  if (view === "training") {
    return <TrainingCenter onBack={() => setView("label")} />;
  }

  if (view === "monitor") {
    return <RunMonitor onBack={() => setView("label")} />;
  }

  return (
    <div style={{ position: "relative" }}>
      <LabelStudio />
      <Space style={{ position: "fixed", right: 16, bottom: 16, zIndex: 1000 }}>
        <Button
          icon={<MonitorOutlined />}
          style={{ boxShadow: "0 2px 8px rgba(0,0,0,0.15)" }}
          onClick={() => setView("monitor")}
        >
          运行监控
        </Button>
        <Button
          id="tour-training-entry"
          icon={<ExperimentOutlined />}
          style={{ boxShadow: "0 2px 8px rgba(0,0,0,0.15)" }}
          onClick={() => setView("training")}
        >
          训练中心
        </Button>
      </Space>
    </div>
  );
}
