import { useState } from "react";
import { setToken } from "../api/client";
import "@fontsource-variable/outfit";
import "./tokengate.css";

/** Shown when the backend requires an access token (remote exposure). */
export default function TokenGate() {
  const [value, setValue] = useState("");
  const [error, setError] = useState(false);

  const submit = () => {
    if (!value.trim()) {
      setError(true);
      return;
    }
    setToken(value.trim());
    window.location.reload();
  };

  return (
    <div className="tg-root">
      <div className="tg-card">
        <div className="tg-mark" aria-hidden>
          <img src="/logo-icon.png" alt="TrainLens logo" width="26" height="26" />
        </div>
        <h1 className="tg-title">输入访问令牌</h1>
        <p className="tg-sub">
          该服务运行在远程模式。令牌显示在服务器启动时的控制台输出中。
        </p>
        <label className="tg-label" htmlFor="tg-input">
          访问令牌
        </label>
        <input
          id="tg-input"
          className={`tg-input${error ? " tg-input-error" : ""}`}
          type="password"
          placeholder="粘贴令牌后回车"
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            setError(false);
          }}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          autoFocus
        />
        {error && <div className="tg-error">令牌不能为空</div>}
        <button type="button" className="tg-btn" onClick={submit}>
          进入
        </button>
      </div>
    </div>
  );
}
