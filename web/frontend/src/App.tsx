import { Button, Space } from "antd";
import { ExperimentOutlined, MonitorOutlined } from "@ant-design/icons";
import { useEffect, useState } from "react";
import LabelStudio from "./pages/LabelStudio";
import TrainingCenter from "./pages/TrainingCenter";
import RunMonitor from "./pages/RunMonitor";
import Welcome from "./pages/Welcome";
import TokenGate from "./pages/TokenGate";
import { useStudio } from "./store/useStudio";

export default function App() {
  const dir = useStudio((s) => s.dir);
  const video = useStudio((s) => s.video);
  const [view, setView] = useState<"label" | "training" | "monitor">("label");
  const [needsAuth, setNeedsAuth] = useState(false);

  useEffect(() => {
    const onUnauthorized = () => setNeedsAuth(true);
    window.addEventListener("xaw:unauthorized", onUnauthorized);
    return () => window.removeEventListener("xaw:unauthorized", onUnauthorized);
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
