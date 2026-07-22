import { Button, Divider, message, Space, Tooltip, Upload } from "antd";
import {
  AimOutlined,
  BlockOutlined,
  BorderOutlined,
  CheckCircleOutlined,
  CopyOutlined,
  DeleteOutlined,
  ExportOutlined,
  FolderOpenOutlined,
  FontSizeOutlined,
  HomeOutlined,
  LeftOutlined,
  LineOutlined,
  NodeIndexOutlined,
  PlusOutlined,
  RightOutlined,
  SaveOutlined,
  SelectOutlined,
  UndoOutlined,
  UploadOutlined,
  VideoCameraOutlined,
} from "@ant-design/icons";
import { useStudio } from "../store/useStudio";
import { uploadFiles } from "../api/client";
import type { ShapeType } from "../types";

const CREATE_TOOLS: { type: ShapeType; label: string; icon: React.ReactNode; key: string }[] = [
  { type: "rectangle", label: "矩形 (R)", icon: <BorderOutlined />, key: "r" },
  { type: "polygon", label: "多边形 (P)", icon: <NodeIndexOutlined />, key: "p" },
  { type: "rotation", label: "旋转框 (O)", icon: <UndoOutlined />, key: "o" },
  { type: "circle", label: "圆形 (C)", icon: <CheckCircleOutlined />, key: "c" },
  { type: "line", label: "直线 (L)", icon: <LineOutlined />, key: "l" },
  { type: "point", label: "点 (T)", icon: <PlusOutlined />, key: "t" },
  { type: "linestrip", label: "折线 (S)", icon: <FontSizeOutlined />, key: "s" },
  { type: "cuboid", label: "立方体 (U)", icon: <BlockOutlined />, key: "u" },
];

interface Props {
  onOpenDir: () => void;
  onOpenVideo: () => void;
  onExport: () => void;
}

export default function Toolbar({ onOpenDir, onOpenVideo, onExport }: Props) {
  const {
    mode,
    createType,
    selected,
    dirty,
    video,
    setMode,
    setCreateType,
    removeShape,
    saveCurrent,
    nextImage,
    prevImage,
    requestFit,
    copyPrevFrame,
    closeSession,
    currentIndex,
    images,
  } = useStudio();

  const totalCount = video ? video.frameCount : images.length;

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "6px 12px",
        background: "#fff",
        borderBottom: "1px solid #f0f0f0",
        flexWrap: "wrap",
      }}
    >
      <Space size={4}>
        <Tooltip title="返回首页">
          <Button icon={<HomeOutlined />} onClick={closeSession} />
        </Tooltip>
        <Tooltip title="打开目录">
          <Button icon={<FolderOpenOutlined />} onClick={onOpenDir}>
            打开目录
          </Button>
        </Tooltip>
        <Tooltip title="打开视频">
          <Button icon={<VideoCameraOutlined />} onClick={onOpenVideo}>
            打开视频
          </Button>
        </Tooltip>
        <Tooltip title="保存 (Ctrl+S)">
          <Button
            icon={<SaveOutlined />}
            type={dirty ? "primary" : "default"}
            onClick={() => saveCurrent()}
            disabled={currentIndex < 0}
          >
            保存
          </Button>
        </Tooltip>
      </Space>

      <Divider type="vertical" />

      <Space size={4}>
        <Tooltip title={video ? "上一帧 (A)" : "上一张 (A)"}>
          <Button icon={<LeftOutlined />} onClick={prevImage} disabled={currentIndex <= 0} />
        </Tooltip>
        <span style={{ minWidth: 90, textAlign: "center", color: "#666" }}>
          {currentIndex >= 0 ? `${currentIndex + 1} / ${totalCount}` : "- / -"}
        </span>
        <Tooltip title={video ? "下一帧 (D)" : "下一张 (D)"}>
          <Button
            icon={<RightOutlined />}
            onClick={nextImage}
            disabled={currentIndex < 0 || currentIndex >= totalCount - 1}
          />
        </Tooltip>
        {video && (
          <Tooltip title="复制上一帧标注到当前帧">
            <Button
              icon={<CopyOutlined />}
              onClick={copyPrevFrame}
              disabled={currentIndex <= 0}
            />
          </Tooltip>
        )}
      </Space>

      <Divider type="vertical" />

      <Space size={4}>
        <Tooltip title="选择/编辑 (V)">
          <Button
            icon={<SelectOutlined />}
            type={mode === "select" ? "primary" : "default"}
            onClick={() => setMode("select")}
          />
        </Tooltip>
        {CREATE_TOOLS.map((t) => (
          <Tooltip key={t.type} title={t.label}>
            <Button
              icon={t.icon}
              type={mode === "create" && createType === t.type ? "primary" : "default"}
              onClick={() => setCreateType(t.type)}
            />
          </Tooltip>
        ))}
      </Space>

      <Divider type="vertical" />

      <Space size={4}>
        <Tooltip title="删除选中 (Del)">
          <Button
            icon={<DeleteOutlined />}
            danger
            disabled={selected === null}
            onClick={() => selected !== null && removeShape(selected)}
          />
        </Tooltip>
        <Tooltip title="适应窗口 (F)">
          <Button icon={<AimOutlined />} onClick={requestFit} />
        </Tooltip>
      </Space>

      <Divider type="vertical" />

      <Space size={4}>
        <Tooltip title="导出为 YOLO/VOC/COCO 等格式">
          <Button
            icon={<ExportOutlined />}
            onClick={onExport}
            disabled={currentIndex < 0 || !!video}
          >
            导出
          </Button>
        </Tooltip>
        <Upload
          multiple
          showUploadList={false}
          beforeUpload={(_, fileList) => {
            uploadFiles(fileList as File[])
              .then(async (r) => {
                message.success(
                  `已上传 ${r.saved} 个文件` +
                    (r.skipped.length ? `，跳过 ${r.skipped.length} 个` : "")
                );
                await useStudio.getState().refreshImages();
              })
              .catch((e) =>
                message.error(
                  `上传失败: ${e?.response?.data?.detail ?? e.message}`
                )
              );
            return false;
          }}
        >
          <Tooltip title="上传图片/标注文件到当前目录">
            <Button icon={<UploadOutlined />} disabled={currentIndex < 0 || !!video}>
              上传
            </Button>
          </Tooltip>
        </Upload>
      </Space>
    </div>
  );
}
