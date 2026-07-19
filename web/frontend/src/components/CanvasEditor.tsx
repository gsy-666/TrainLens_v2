import { useCallback, useEffect, useRef, useState } from "react";
import { Stage, Layer, Image as KonvaImage, Line, Circle, Rect } from "react-konva";
import type Konva from "konva";
import type { KonvaEventObject } from "konva/lib/Node";
import { useStudio } from "../store/useStudio";
import { imageUrl, videoFrameUrl } from "../api/client";
import { labelColor, withAlpha } from "../utils/colors";
import type { Point, Shape, ShapeType } from "../types";

interface ViewState {
  scale: number;
  x: number;
  y: number;
}

interface Props {
  onFinishDraft: (points: Point[], type: ShapeType) => void;
  onSamPrompt: (start: Point, end: Point) => void;
}

const HANDLE_RADIUS = 4;
const MIN_DRAG_PX = 3; // min drag distance (image px) to finish a drag shape
const SAM_CLICK_PX = 3; // below this a SAM prompt counts as a point

export default function CanvasEditor({ onFinishDraft, onSamPrompt }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const [size, setSize] = useState({ w: 800, h: 600 });
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [view, setView] = useState<ViewState>({ scale: 1, x: 0, y: 0 });

  // drawing draft state
  const [draftPts, setDraftPts] = useState<Point[]>([]);
  const [previewPt, setPreviewPt] = useState<Point | null>(null);
  const [dragging, setDragging] = useState(false);
  const [rotPhase, setRotPhase] = useState<0 | 1>(0); // rotation: 0=edge, 1=extent
  const panRef = useRef<{ startX: number; startY: number; vx: number; vy: number; moved: boolean } | null>(null);

  // SAM prompt state
  const [samStart, setSamStart] = useState<Point | null>(null);
  const [samCur, setSamCur] = useState<Point | null>(null);

  const {
    images,
    video,
    currentIndex,
    shapes,
    hidden,
    selected,
    mode,
    createType,
    fitRequest,
    samMode,
    setSelected,
    updateShape,
    setImageSize,
    beginShapeEdit,
    endShapeEdit,
  } = useStudio();

  const currentFile =
    currentIndex >= 0
      ? video
        ? `frame-${currentIndex}`
        : images[currentIndex]?.filename
      : null;

  // ---- container size ----------------------------------------------------
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      setSize({ w: el.clientWidth, h: el.clientHeight });
    });
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  // ---- image loading ------------------------------------------------------
  useEffect(() => {
    if (!currentFile) {
      setImg(null);
      return;
    }
    let cancelled = false;
    const image = new window.Image();
    image.src = video ? videoFrameUrl(currentIndex) : imageUrl(currentFile);
    image.onload = () => {
      if (cancelled) return;
      setImg(image);
      setImageSize(image.naturalWidth, image.naturalHeight);
    };
    image.onerror = () => {
      if (!cancelled) setImg(null);
    };
    return () => {
      cancelled = true;
      image.onload = null;
      image.onerror = null;
    };
  }, [currentFile, video, currentIndex, setImageSize]);

  // ---- fit to window -------------------------------------------------------
  const fit = useCallback(() => {
    if (!img || size.w === 0) return;
    const scale = Math.min(size.w / img.naturalWidth, size.h / img.naturalHeight) * 0.96;
    setView({
      scale,
      x: (size.w - img.naturalWidth * scale) / 2,
      y: (size.h - img.naturalHeight * scale) / 2,
    });
  }, [img, size]);

  useEffect(() => {
    fit();
  }, [fit, fitRequest, currentFile]);

  // cancel draft when switching image/mode/type
  useEffect(() => {
    setDraftPts([]);
    setPreviewPt(null);
    setDragging(false);
    setRotPhase(0);
    setSamStart(null);
    setSamCur(null);
  }, [currentFile, mode, createType, samMode]);

  // ---- helpers -------------------------------------------------------------
  const toImagePos = useCallback((): Point | null => {
    const stage = stageRef.current;
    if (!stage) return null;
    const p = stage.getPointerPosition();
    if (!p) return null;
    return [(p.x - view.x) / view.scale, (p.y - view.y) / view.scale];
  }, [view]);

  const finishDraft = useCallback(
    (pts: Point[]) => {
      setDraftPts([]);
      setPreviewPt(null);
      setDragging(false);
      setRotPhase(0);
      onFinishDraft(pts, createType);
    },
    [onFinishDraft, createType]
  );

  // cancel draft on Escape (global handler lives in LabelStudio; expose via ref)
  const cancelDraft = useCallback(() => {
    setDraftPts([]);
    setPreviewPt(null);
    setDragging(false);
    setRotPhase(0);
  }, []);
  useEffect(() => {
    (window as unknown as { __cancelCanvasDraft?: () => void }).__cancelCanvasDraft =
      cancelDraft;
    return () => {
      delete (window as unknown as { __cancelCanvasDraft?: () => void })
        .__cancelCanvasDraft;
    };
  }, [cancelDraft]);

  // ---- wheel zoom -----------------------------------------------------------
  const onWheel = useCallback(
    (e: KonvaEventObject<WheelEvent>) => {
      e.evt.preventDefault();
      const stage = stageRef.current;
      if (!stage) return;
      const pointer = stage.getPointerPosition();
      if (!pointer) return;
      const factor = e.evt.deltaY > 0 ? 1 / 1.15 : 1.15;
      const newScale = Math.min(50, Math.max(0.02, view.scale * factor));
      const imgX = (pointer.x - view.x) / view.scale;
      const imgY = (pointer.y - view.y) / view.scale;
      setView({
        scale: newScale,
        x: pointer.x - imgX * newScale,
        y: pointer.y - imgY * newScale,
      });
    },
    [view]
  );

  // ---- mouse handlers --------------------------------------------------------
  const onMouseDown = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      const p = toImagePos();
      if (!p) return;

      // SAM prompt mode: click = point prompt, drag = box prompt
      if (samMode && !video && e.evt.button === 0) {
        setSamStart(p);
        setSamCur(p);
        return;
      }

      if (mode === "select") {
        // click on empty background -> start panning (click without move = deselect)
        const target = e.target;
        const stage = stageRef.current;
        const isEmpty =
          target === stage || (target.className === "Image" && !e.evt.shiftKey);
        if (isEmpty && e.evt.button === 0) {
          const pointer = stage!.getPointerPosition()!;
          panRef.current = {
            startX: pointer.x,
            startY: pointer.y,
            vx: view.x,
            vy: view.y,
            moved: false,
          };
        }
        return;
      }

      // create mode
      if (e.evt.button !== 0) return;
      switch (createType) {
        case "point":
          finishDraft([p]);
          break;
        case "rectangle":
        case "line":
        case "circle":
        case "cuboid":
          setDraftPts([p]);
          setDragging(true);
          break;
        case "polygon":
        case "linestrip":
          if (draftPts.length >= 1) {
            const [fx, fy] = draftPts[0];
            const dist = Math.hypot(p[0] - fx, p[1] - fy) * view.scale;
            if (createType === "polygon" && draftPts.length >= 3 && dist < 10) {
              finishDraft(draftPts);
              return;
            }
          }
          setDraftPts((pts) => [...pts, p]);
          break;
        case "rotation":
          if (rotPhase === 0) {
            setDraftPts([p]);
            setDragging(true);
          } else {
            // finish with current preview extent
            const quad = rotationQuad(draftPts, p);
            if (quad) finishDraft(quad);
          }
          break;
      }
    },
    [mode, createType, draftPts, rotPhase, view, toImagePos, finishDraft, samMode, video]
  );

  const onMouseMove = useCallback(() => {
    if (panRef.current) {
      const stage = stageRef.current;
      if (!stage) return;
      const pointer = stage.getPointerPosition();
      if (!pointer) return;
      const dx = pointer.x - panRef.current.startX;
      const dy = pointer.y - panRef.current.startY;
      if (Math.abs(dx) + Math.abs(dy) > 3) panRef.current.moved = true;
      if (panRef.current.moved) {
        setView((v) => ({ ...v, x: panRef.current!.vx + dx, y: panRef.current!.vy + dy }));
      }
      return;
    }
    const p = toImagePos();
    if (!p) return;
    if (samMode && samStart) {
      setSamCur(p);
      return;
    }
    setPreviewPt(p);

    if (mode === "create" && dragging && draftPts.length === 1) {
      if (createType === "rotation" && rotPhase === 0) {
        // first edge being dragged
        setDraftPts((pts) => [pts[0]]);
      }
    }
  }, [mode, dragging, draftPts.length, createType, rotPhase, toImagePos]);

  const onMouseUp = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (samMode && samStart) {
        const p = toImagePos();
        const a = samStart;
        setSamStart(null);
        setSamCur(null);
        if (p) onSamPrompt(a, p);
        return;
      }
      if (panRef.current) {
        const wasClick = !panRef.current.moved;
        panRef.current = null;
        if (wasClick && mode === "select") setSelected(null);
        return;
      }
      if (mode !== "create" || !dragging) return;
      const p = toImagePos();
      if (!p) return;
      const [start] = draftPts;
      if (!start) return;
      const dist = Math.hypot(p[0] - start[0], p[1] - start[1]);
      if (dist < MIN_DRAG_PX) {
        if (createType === "rotation") return; // keep waiting for a real edge
        setDragging(false);
        setDraftPts([]);
        return;
      }
      switch (createType) {
        case "rectangle": {
          const [x1, y1] = [Math.min(start[0], p[0]), Math.min(start[1], p[1])];
          const [x2, y2] = [Math.max(start[0], p[0]), Math.max(start[1], p[1])];
          finishDraft([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
          ]);
          break;
        }
        case "line":
          finishDraft([start, p]);
          break;
        case "circle":
          finishDraft([start, p]);
          break;
        case "cuboid": {
          const quad = rectQuad(start, p);
          const dv: Point = [24, -24]; // desktop default depth vector
          finishDraft([...quad, ...quad.map(([x, y]) => [x + dv[0], y + dv[1]] as Point)]);
          break;
        }
        case "rotation":
          // edge defined, move to extent phase
          setDraftPts([start, p]);
          setDragging(false);
          setRotPhase(1);
          break;
      }
    },
    [mode, dragging, draftPts, createType, toImagePos, finishDraft, setSelected, samMode, samStart, onSamPrompt]
  );

  const onDblClick = useCallback(() => {
    if (mode === "create" && (createType === "polygon" || createType === "linestrip")) {
      const min = createType === "polygon" ? 3 : 2;
      if (draftPts.length >= min) finishDraft(draftPts);
    }
  }, [mode, createType, draftPts, finishDraft]);

  // ---- shape commit helpers ---------------------------------------------------
  const moveShape = useCallback(
    (index: number, dx: number, dy: number) => {
      const s = shapes[index];
      updateShape(index, {
        ...s,
        points: s.points.map(([x, y]) => [x + dx, y + dy] as Point),
      });
    },
    [shapes, updateShape]
  );

  const moveVertex = useCallback(
    (index: number, vi: number, pos: Point) => {
      const s = shapes[index];
      const pts = s.points.map((pt, i) => (i === vi ? pos : pt));
      updateShape(index, { ...s, points: pts }, { history: false });
    },
    [shapes, updateShape]
  );

  // ---- render ---------------------------------------------------------------
  const renderShape = (s: Shape, i: number) => {
    if (hidden[i]) return null;
    const color = labelColor(s.label);
    const isSel = selected === i;
    const common = {
      key: i,
      stroke: isSel ? "#ffffff" : color,
      strokeWidth: 2,
      strokeScaleEnabled: false,
      fill: withAlpha(color, isSel ? 0.35 : 0.15),
      listening: mode === "select",
      onClick: () => mode === "select" && setSelected(i),
      draggable: mode === "select",
      onDragEnd: (e: KonvaEventObject<DragEvent>) => {
        const node = e.target;
        moveShape(i, node.x(), node.y());
        node.position({ x: 0, y: 0 });
      },
    };
    const flat = s.points.flat();

    switch (s.shape_type) {
      case "point": {
        const [cx, cy] = s.points[0] ?? [0, 0];
        return (
          <Circle
            {...common}
            x={cx}
            y={cy}
            radius={4}
            fill={color}
            onDragStart={() => beginShapeEdit()}
            onDragMove={(e) => moveVertex(i, 0, [e.target.x(), e.target.y()])}
            onDragEnd={() => endShapeEdit()}
          />
        );
      }
      case "circle": {
        const [c, edge] = s.points;
        if (!c || !edge) return null;
        const r = Math.hypot(c[0] - edge[0], c[1] - edge[1]);
        return (
          <Circle
            {...common}
            x={c[0]}
            y={c[1]}
            radius={r}
            onDragEnd={(e) => {
              const node = e.target;
              moveShape(i, node.x(), node.y());
              node.position({ x: 0, y: 0 });
            }}
          />
        );
      }
      case "polygon":
      case "rectangle":
      case "rotation":
        return <Line {...common} points={flat} closed perfectDrawEnabled={false} />;
      case "cuboid": {
        if (s.points.length < 8) {
          return <Line {...common} points={flat} closed perfectDrawEnabled={false} />;
        }
        const front = s.points.slice(0, 4).flat();
        const back = s.points.slice(4, 8).flat();
        return (
          <>
            <Line {...common} key={`${i}-f`} points={front} closed perfectDrawEnabled={false} />
            <Line {...common} key={`${i}-b`} points={back} closed perfectDrawEnabled={false} />
            {[0, 1, 2, 3].map((k) => (
              <Line
                {...common}
                key={`${i}-e${k}`}
                points={[...s.points[k], ...s.points[k + 4]]}
                fillEnabled={false}
                perfectDrawEnabled={false}
              />
            ))}
          </>
        );
      }
      case "line":
      case "linestrip":
      default:
        return <Line {...common} points={flat} closed={false} perfectDrawEnabled={false} />;
    }
  };

  const renderHandles = () => {
    if (selected === null || mode !== "select") return null;
    const s = shapes[selected];
    if (!s || hidden[selected]) return null;
    return s.points.map(([x, y], vi) => (
      <Circle
        key={vi}
        x={x}
        y={y}
        radius={HANDLE_RADIUS}
        fill="#00ff00"
        stroke="#ffffff"
        strokeWidth={1}
        strokeScaleEnabled={false}
        draggable
        onDragStart={() => beginShapeEdit()}
        onDragMove={(e) => moveVertex(selected, vi, [e.target.x(), e.target.y()])}
        onDragEnd={() => endShapeEdit()}
      />
    ));
  };

  const renderDraft = () => {
    if (mode !== "create") return null;
    const pts = [...draftPts];
    if (previewPt) pts.push(previewPt);
    if (pts.length === 0) return null;
    const color = "#ff4d4f";
    const common = {
      stroke: color,
      strokeWidth: 2,
      dash: [6, 4],
      strokeScaleEnabled: false,
      listening: false,
    };
    switch (createType) {
      case "rectangle": {
        if (pts.length < 2) return null;
        const [a, b] = pts;
        return (
          <Rect
            {...common}
            x={Math.min(a[0], b[0])}
            y={Math.min(a[1], b[1])}
            width={Math.abs(a[0] - b[0])}
            height={Math.abs(a[1] - b[1])}
          />
        );
      }
      case "circle": {
        if (pts.length < 2) return null;
        const [c, e] = pts;
        return (
          <Circle {...common} x={c[0]} y={c[1]} radius={Math.hypot(c[0] - e[0], c[1] - e[1])} />
        );
      }
      case "cuboid": {
        if (pts.length < 2) return null;
        const quad = rectQuad(pts[0], pts[1]);
        const dv: Point = [24, -24];
        const back = quad.map(([x, y]) => [x + dv[0], y + dv[1]] as Point);
        return (
          <>
            <Line {...common} points={quad.flat()} closed />
            <Line {...common} points={back.flat()} closed />
            {[0, 1, 2, 3].map((k) => (
              <Line key={k} {...common} points={[...quad[k], ...back[k]]} />
            ))}
          </>
        );
      }
      case "rotation": {
        if (rotPhase === 1 && previewPt && draftPts.length === 2) {
          const quad = rotationQuad(draftPts, previewPt);
          if (quad) return <Line {...common} points={quad.flat()} closed />;
        }
        return <Line {...common} points={pts.flat()} />;
      }
      case "polygon":
        return (
          <>
            <Line {...common} points={pts.flat()} closed={false} />
            {draftPts.map(([x, y], i) => (
              <Circle key={i} x={x} y={y} radius={3} fill={color} listening={false} />
            ))}
          </>
        );
      default:
        return <Line {...common} points={pts.flat()} />;
    }
  };

  const renderSamPreview = () => {
    if (!samMode || !samStart) return null;
    const color = "#2563eb";
    const common = {
      stroke: color,
      strokeWidth: 2,
      dash: [6, 4],
      strokeScaleEnabled: false,
      listening: false,
    };
    const isBox =
      samCur &&
      Math.hypot(samCur[0] - samStart[0], samCur[1] - samStart[1]) > SAM_CLICK_PX;
    if (isBox && samCur) {
      return (
        <Rect
          {...common}
          x={Math.min(samStart[0], samCur[0])}
          y={Math.min(samStart[1], samCur[1])}
          width={Math.abs(samCur[0] - samStart[0])}
          height={Math.abs(samCur[1] - samStart[1])}
        />
      );
    }
    // point marker: small cross
    const [cx, cy] = samStart;
    const r = 6;
    return (
      <>
        <Circle x={cx} y={cy} radius={2.5} fill={color} listening={false} />
        <Line {...common} points={[cx - r, cy, cx + r, cy]} dash={undefined} />
        <Line {...common} points={[cx, cy - r, cx, cy + r]} dash={undefined} />
      </>
    );
  };

  const cursor = samMode
    ? "crosshair"
    : mode === "create"
      ? "crosshair"
      : panRef.current?.moved
        ? "grabbing"
        : "default";

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "100%", overflow: "hidden", background: "#1f1f1f", cursor }}
    >
      <Stage
        ref={stageRef}
        width={size.w}
        height={size.h}
        x={view.x}
        y={view.y}
        scaleX={view.scale}
        scaleY={view.scale}
        onWheel={onWheel}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onDblClick={onDblClick}
      >
        <Layer listening={false}>{img && <KonvaImage key={currentFile} image={img} />}</Layer>
        <Layer>
          {shapes.map(renderShape)}
          {renderHandles()}
          {renderDraft()}
          {renderSamPreview()}
        </Layer>
      </Stage>
    </div>
  );
}

/** Normalized axis-aligned rectangle as 4 corner points. */
function rectQuad(a: Point, b: Point): Point[] {
  const x1 = Math.min(a[0], b[0]);
  const y1 = Math.min(a[1], b[1]);
  const x2 = Math.max(a[0], b[0]);
  const y2 = Math.max(a[1], b[1]);
  return [
    [x1, y1],
    [x2, y1],
    [x2, y2],
    [x1, y2],
  ];
}

/** Build a rotated rectangle (4 points) from an edge and an extent point. */
function rotationQuad(edge: Point[], extent: Point | null): Point[] | null {
  if (edge.length !== 2 || !extent) return null;
  const [[x1, y1], [x2, y2]] = edge;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy);
  if (len < 1e-6) return null;
  // perpendicular unit vector
  const nx = -dy / len;
  const ny = dx / len;
  // signed distance of extent point from edge line
  const w = (extent[0] - x1) * nx + (extent[1] - y1) * ny;
  return [
    [x1, y1],
    [x2, y2],
    [x2 + nx * w, y2 + ny * w],
    [x1 + nx * w, y1 + ny * w],
  ];
}
