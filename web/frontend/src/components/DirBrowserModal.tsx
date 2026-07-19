import { useCallback, useEffect, useRef, useState } from "react";
import { Breadcrumb, Button, Empty, Input, List, Modal, Spin, Tag } from "antd";
import {
  ArrowUpOutlined,
  FileImageOutlined,
  FolderOutlined,
  HddOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import { fsList, type FsDirEntry } from "../api/client";

interface Props {
  open: boolean;
  initialPath?: string;
  title?: string;
  fileExtensions?: string[]; // when set, files are listed & selectable
  onSelect: (path: string) => void;
  onCancel: () => void;
}

export default function DirBrowserModal({
  open,
  initialPath,
  title,
  fileExtensions,
  onSelect,
  onCancel,
}: Props) {
  const [current, setCurrent] = useState<string>(""); // "" = roots view
  const [roots, setRoots] = useState<string[]>([]);
  const [dirs, setDirs] = useState<FsDirEntry[]>([]);
  const [files, setFiles] = useState<{ name: string; path: string }[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [parent, setParent] = useState<string | null>(null);
  const [hasImages, setHasImages] = useState(false);
  const [loading, setLoading] = useState(false);
  const [pathInput, setPathInput] = useState("");
  const [error, setError] = useState<string | null>(null);

  const fileMode = !!fileExtensions && fileExtensions.length > 0;
  // stable primitive key — callers often pass inline array literals, whose
  // changing identity would retrigger the mount effect below on every render
  const extsKey = fileMode ? fileExtensions!.join(",") : "";

  const navigate = useCallback(
    async (path: string) => {
      setLoading(true);
      setError(null);
      setSelectedFile(null);
      try {
        const res = await fsList(path, extsKey || undefined);
        if (res.roots) {
          setRoots(res.roots);
          setCurrent("");
          setDirs([]);
          setFiles([]);
          setParent(null);
          setHasImages(false);
          setPathInput("");
        } else {
          setRoots([]);
          setCurrent(res.path);
          setDirs(res.dirs);
          setFiles(res.files ?? []);
          setParent(res.parent);
          setHasImages(res.has_images);
          setPathInput(res.path);
        }
      } catch (e) {
        const err = e as { response?: { data?: { detail?: string } }; message: string };
        setError(err.response?.data?.detail ?? err.message);
      } finally {
        setLoading(false);
      }
    },
    [extsKey]
  );

  const navigateRef = useRef(navigate);
  navigateRef.current = navigate;

  // navigate only when the dialog is (re)opened or initialPath changes —
  // never because unrelated parent re-renders rebuilt the callback
  const openedRef = useRef(false);
  useEffect(() => {
    if (open && !openedRef.current) {
      openedRef.current = true;
      navigateRef.current(initialPath || "");
    } else if (!open) {
      openedRef.current = false;
    }
  }, [open, initialPath]);

  const goUp = () => {
    if (parent !== null) navigate(parent);
    else navigate("");
  };

  const atRoots = current === "";

  return (
    <Modal
      open={open}
      title={title ?? "选择图片目录"}
      width={560}
      onCancel={onCancel}
      footer={[
        <Button key="cancel" onClick={onCancel}>
          取消
        </Button>,
        <Button
          key="ok"
          type="primary"
          disabled={fileMode ? !selectedFile : atRoots}
          onClick={() => onSelect(fileMode ? selectedFile! : current)}
        >
          {fileMode ? "选择此文件" : `选择此目录${hasImages ? "" : "（无图片）"}`}
        </Button>,
      ]}
    >
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <Input
          value={pathInput}
          onChange={(e) => setPathInput(e.target.value)}
          onPressEnter={() => navigate(pathInput.trim())}
          placeholder="输入或粘贴路径后回车跳转"
        />
        <Button onClick={() => navigate(pathInput.trim())}>转到</Button>
        <Button icon={<ArrowUpOutlined />} onClick={goUp} disabled={atRoots && roots.length > 0} title="上一级" />
        <Button icon={<ReloadOutlined />} onClick={() => navigate(current)} title="刷新" />
      </div>

      <Breadcrumb
        style={{ marginBottom: 8 }}
        items={[
          {
            title: (
              <a onClick={() => navigate("")}>此电脑</a>
            ),
          },
          ...(current
            ? current
                .split(/[/\\]/)
                .filter(Boolean)
                .map((seg, i, arr) => ({
                  title:
                    i === arr.length - 1 ? (
                      seg
                    ) : (
                      <a
                        onClick={() =>
                          navigate(arr.slice(0, i + 1).join("\\") + "\\")
                        }
                      >
                        {seg}
                      </a>
                    ),
                }))
            : []),
        ]}
      />

      <div
        style={{
          height: 320,
          overflow: "auto",
          border: "1px solid #f0f0f0",
          borderRadius: 6,
        }}
      >
        {loading ? (
          <div style={{ textAlign: "center", paddingTop: 100 }}>
            <Spin />
          </div>
        ) : error ? (
          <Empty description={error} style={{ marginTop: 80 }} />
        ) : atRoots ? (
          <List
            size="small"
            dataSource={roots}
            renderItem={(r) => (
              <List.Item style={{ cursor: "pointer", padding: "8px 16px" }} onClick={() => navigate(r)}>
                <HddOutlined style={{ marginRight: 8 }} />
                {r}
              </List.Item>
            )}
          />
        ) : dirs.length === 0 && files.length === 0 ? (
          <Empty description={fileMode ? "没有子目录或匹配文件" : "没有子目录"} style={{ marginTop: 80 }} />
        ) : (
          <List
            size="small"
            dataSource={[
              ...dirs.map((d) => ({ kind: "dir" as const, ...d })),
              ...files.map((f) => ({ kind: "file" as const, ...f, has_images: false })),
            ]}
            renderItem={(d) =>
              d.kind === "dir" ? (
                <List.Item
                  style={{ cursor: "pointer", padding: "6px 16px" }}
                  onClick={() => navigate(d.path)}
                >
                  <FolderOutlined style={{ marginRight: 8, color: "#faad14" }} />
                  <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis" }}>
                    {d.name}
                  </span>
                  {d.has_images && (
                    <Tag icon={<FileImageOutlined />} color="green" style={{ marginRight: 0 }}>
                      有图片
                    </Tag>
                  )}
                </List.Item>
              ) : (
                <List.Item
                  style={{
                    cursor: "pointer",
                    padding: "6px 16px",
                    background: selectedFile === d.path ? "#e6f4ff" : undefined,
                  }}
                  onClick={() => setSelectedFile(d.path)}
                  onDoubleClick={() => onSelect(d.path)}
                >
                  <FileImageOutlined style={{ marginRight: 8, color: "#722ed1" }} />
                  <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis" }}>
                    {d.name}
                  </span>
                </List.Item>
              )
            }
          />
        )}
      </div>
      <div style={{ marginTop: 6, fontSize: 12, color: "#999" }}>
        {fileMode
          ? "单击进入目录，单击选中文件，双击文件直接打开"
          : "单击进入目录；选中后点「选择此目录」打开其中的图片"}
      </div>
    </Modal>
  );
}
