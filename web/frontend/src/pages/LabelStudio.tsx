import { useCallback, useEffect, useState } from "react";
import { Layout, message, Splitter } from "antd";
import Toolbar from "../components/Toolbar";
import FileList from "../components/FileList";
import LabelList from "../components/LabelList";
import LabelDialog, { type LabelFormValue } from "../components/LabelDialog";
import DirBrowserModal from "../components/DirBrowserModal";
import ExportDialog from "../components/ExportDialog";
import CanvasEditor from "../components/CanvasEditor";
import ModelPanel from "../components/ModelPanel";
import GuidedTour, { useGuidedTour, type GuidedTourStep } from "../components/GuidedTour";
import * as api from "../api/client";
import { useStudio } from "../store/useStudio";
import { readPanelWidth, savePanelWidth } from "../utils/panelStorage";
import type { Point, Shape, ShapeType } from "../types";

const LS_LEFT_WIDTH_KEY = "xaw_ls_left_width";
const LS_RIGHT_WIDTH_KEY = "xaw_ls_right_width";

const TOUR_STEPS: GuidedTourStep[] = [
  {
    targetId: "tour-file-list",
    title: "文件列表",
    description: "在这里切换、搜索和筛选图片，带绿勾的表示已经标注过。",
  },
  {
    targetId: "tour-canvas-toolbar",
    title: "标注工具栏",
    description: "选择矩形、多边形等工具后，在中间画布上拖拽鼠标就能框出目标。",
  },
  {
    targetId: "tour-label-list",
    title: "标签列表",
    description: "当前图片的所有标注都列在这里，可以点击选中、修改标签或删除。",
  },
  {
    targetId: "tour-model-panel",
    title: "AI 自动标注",
    description: "先选择并「加载模型」，再点「运行」即可自动预标注，之后人工修正就好。",
  },
  {
    targetId: "tour-save-btn",
    title: "保存标注",
    description: "标注完记得保存，也可以随时按 Ctrl+S。",
  },
  {
    targetId: "tour-training-entry",
    title: "去训练中心",
    description: "标注积累够了，点右下角「训练中心」就能把标注训练成自己的模型。",
  },
];

export default function LabelStudio() {
  const {
    dir,
    openDir,
    openVideo,
    saveCurrent,
    nextImage,
    prevImage,
    addShape,
    updateShape,
    removeShape,
    shapes,
    images,
    selected,
    setSelected,
    setMode,
    setCreateType,
    requestFit,
    currentIndex,
  } = useStudio();

  const [openDirVisible, setOpenDirVisible] = useState(false);
  const [openVideoVisible, setOpenVideoVisible] = useState(false);
  const [exportVisible, setExportVisible] = useState(false);
  const [draft, setDraft] = useState<{ points: Point[]; type: ShapeType } | null>(null);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const tour = useGuidedTour("xaw_tour_label_seen");
  // 左右栏宽度记忆：defaultSize 只在挂载时读一次，拖拽过程由 Splitter 内部维护（非受控）
  const [leftDefault] = useState(() => readPanelWidth(LS_LEFT_WIDTH_KEY, 220));
  const [rightDefault] = useState(() => readPanelWidth(LS_RIGHT_WIDTH_KEY, 280));

  const handleSelectDir = useCallback(
    async (path: string) => {
      try {
        await openDir(path);
        setOpenDirVisible(false);
      } catch (e) {
        message.error(`打开失败: ${(e as Error).message}`);
      }
    },
    [openDir]
  );

  const handleSelectVideo = useCallback(
    async (path: string) => {
      try {
        await openVideo(path);
        setOpenVideoVisible(false);
        message.success("视频已打开，A/D 或滑杆切换帧", 3);
      } catch (e) {
        const err = e as { response?: { data?: { detail?: string } } } & Error;
        message.error(`打开视频失败: ${err.response?.data?.detail ?? err.message}`);
      }
    },
    [openVideo]
  );

  // ---- label dialog handling -------------------------------------------------
  const handleFinishDraft = useCallback((points: Point[], type: ShapeType) => {
    setDraft({ points, type });
  }, []);

  // ---- SAM prompt handling ----------------------------------------------------
  const handleSamPrompt = useCallback(
    async (a: Point, b: Point) => {
      const file = images[currentIndex]?.filename;
      if (!file) return;
      const isPoint = Math.hypot(b[0] - a[0], b[1] - a[1]) < 3;
      const marks = isPoint
        ? [{ type: "point" as const, data: a, label: 1 }]
        : [
            {
              type: "rectangle" as const,
              data: [
                Math.min(a[0], b[0]),
                Math.min(a[1], b[1]),
                Math.max(a[0], b[0]),
                Math.max(a[1], b[1]),
              ],
            },
          ];
      const hide = message.loading("SAM 推理中…", 0);
      try {
        const res = await api.predictSam(file, marks);
        const polygons = res.shapes.filter((s) => s.points.length >= 3);
        if (polygons.length === 0) {
          message.warning("未生成掩码，换个位置或拖个框试试");
          return;
        }
        const area = (pts: [number, number][]) => {
          let s = 0;
          for (let i = 0; i < pts.length; i++) {
            const [x1, y1] = pts[i];
            const [x2, y2] = pts[(i + 1) % pts.length];
            s += x1 * y2 - x2 * y1;
          }
          return Math.abs(s / 2);
        };
        const best = polygons.reduce((m, s) =>
          area(s.points) > area(m.points) ? s : m
        );
        setDraft({
          points: best.points.map((p) => [p[0], p[1]] as Point),
          type: "polygon",
        });
      } catch (e) {
        const err = e as { response?: { data?: { detail?: string } }; message: string };
        message.error(`SAM 推理失败: ${err.response?.data?.detail ?? err.message}`);
      } finally {
        hide();
      }
    },
    [images, currentIndex]
  );

  const confirmDraft = useCallback(
    (v: LabelFormValue) => {
      if (!draft) return;
      const shape: Shape = {
        label: v.label,
        points: draft.points,
        shape_type: draft.type,
        group_id: v.group_id,
        description: v.description,
        difficult: v.difficult,
        flags: {},
        attributes: {},
        kie_linking: [],
      };
      if (draft.type === "rotation" && draft.points.length >= 2) {
        const [p1, p2] = draft.points;
        shape.direction = Math.atan2(p2[1] - p1[1], p2[0] - p1[0]);
      }
      if (draft.type === "cuboid" && draft.points.length >= 8) {
        shape.cuboid3d = {
          depth_vector: [
            draft.points[4][0] - draft.points[0][0],
            draft.points[4][1] - draft.points[0][1],
          ],
          mode: "from_rectangle",
          source: "manual",
        };
      }
      addShape(shape);
      setDraft(null);
    },
    [draft, addShape]
  );

  const confirmEdit = useCallback(
    (v: LabelFormValue) => {
      if (editingIndex === null) return;
      const s = shapes[editingIndex];
      updateShape(editingIndex, {
        ...s,
        label: v.label,
        group_id: v.group_id,
        description: v.description,
        difficult: v.difficult,
      });
      setEditingIndex(null);
    },
    [editingIndex, shapes, updateShape]
  );

  // ---- global shortcuts -------------------------------------------------------
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      const typing =
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable;

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        saveCurrent().then((ok) => ok && message.success("已保存", 1));
        return;
      }
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === "z") {
        e.preventDefault();
        useStudio.getState().undo();
        return;
      }
      if (
        (e.ctrlKey || e.metaKey) &&
        (e.key.toLowerCase() === "y" || (e.shiftKey && e.key.toLowerCase() === "z"))
      ) {
        e.preventDefault();
        useStudio.getState().redo();
        return;
      }
      if (typing) return;

      const cancelDraft = (window as unknown as { __cancelCanvasDraft?: () => void })
        .__cancelCanvasDraft;

      switch (e.key.toLowerCase()) {
        case "a":
          prevImage();
          break;
        case "d":
          nextImage();
          break;
        case "v":
          setMode("select");
          break;
        case "r":
          setCreateType("rectangle");
          break;
        case "p":
          setCreateType("polygon");
          break;
        case "o":
          setCreateType("rotation");
          break;
        case "c":
          setCreateType("circle");
          break;
        case "l":
          setCreateType("line");
          break;
        case "t":
          setCreateType("point");
          break;
        case "u":
          setCreateType("cuboid");
          break;
        case "f":
          requestFit();
          break;
        case "delete":
        case "backspace":
          if (selected !== null) removeShape(selected);
          break;
        case "escape":
          cancelDraft?.();
          setSelected(null);
          break;
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [
    saveCurrent,
    nextImage,
    prevImage,
    selected,
    removeShape,
    setMode,
    setCreateType,
    requestFit,
    setSelected,
  ]);

  const editingShape = editingIndex !== null ? shapes[editingIndex] : null;

  return (
    <Layout style={{ height: "100vh" }}>
      <Toolbar
        onOpenDir={() => setOpenDirVisible(true)}
        onOpenVideo={() => setOpenVideoVisible(true)}
        onExport={() => setExportVisible(true)}
        onOpenTour={tour.openTour}
      />
      <Layout>
        <Splitter
          style={{ height: "100%" }}
          onResize={(sizes) => {
            savePanelWidth(LS_LEFT_WIDTH_KEY, sizes[0]);
            savePanelWidth(LS_RIGHT_WIDTH_KEY, sizes[2]);
          }}
        >
          <Splitter.Panel defaultSize={leftDefault} min={160} max={480}>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                height: "100%",
                background: "#fff",
                borderRight: "1px solid #f0f0f0",
              }}
            >
              <div id="tour-file-list" style={{ flex: 1, minHeight: 0 }}>
                <FileList />
              </div>
              <div id="tour-model-panel">
                <ModelPanel />
              </div>
            </div>
          </Splitter.Panel>
          {/* 画布面板：不设 defaultSize，自动吃掉剩余宽度。
              注意不能给它 resizable={false}——antd Splitter 里一条拖拽柄要求
              相邻两个面板都可调，中间设 false 会把左右两条柄一起锁死 */}
          <Splitter.Panel min={200}>
            <div style={{ position: "relative", height: "100%" }}>
              {currentIndex >= 0 ? (
                <CanvasEditor onFinishDraft={handleFinishDraft} onSamPrompt={handleSamPrompt} />
              ) : (
                <div
                  style={{
                    height: "100%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "#999",
                  }}
                >
                  请选择或打开一个图片目录
                </div>
              )}
            </div>
          </Splitter.Panel>
          <Splitter.Panel defaultSize={rightDefault} min={220} max={560}>
            <div
              id="tour-label-list"
              style={{ height: "100%", background: "#fff", borderLeft: "1px solid #f0f0f0" }}
            >
              <LabelList onEditLabel={(i) => setEditingIndex(i)} />
            </div>
          </Splitter.Panel>
        </Splitter>
      </Layout>

      <GuidedTour steps={TOUR_STEPS} open={tour.open} onClose={tour.closeTour} />

      <LabelDialog
        open={draft !== null}
        title="输入标签"
        onOk={confirmDraft}
        onCancel={() => setDraft(null)}
      />
      <LabelDialog
        open={editingIndex !== null}
        title="编辑标签"
        initial={
          editingShape
            ? {
                label: editingShape.label,
                group_id: editingShape.group_id ?? null,
                description: editingShape.description ?? "",
                difficult: editingShape.difficult ?? false,
              }
            : undefined
        }
        onOk={confirmEdit}
        onCancel={() => setEditingIndex(null)}
      />

      <DirBrowserModal
        open={openDirVisible}
        initialPath={dir ?? undefined}
        onSelect={handleSelectDir}
        onCancel={() => setOpenDirVisible(false)}
      />

      <ExportDialog open={exportVisible} onCancel={() => setExportVisible(false)} />

      <DirBrowserModal
        open={openVideoVisible}
        title="选择视频文件"
        fileExtensions={[".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"]}
        onSelect={handleSelectVideo}
        onCancel={() => setOpenVideoVisible(false)}
      />
    </Layout>
  );
}
