import { create } from "zustand";
import * as api from "../api/client";
import type { ImageInfo, Shape, ShapeType } from "../types";

export type EditorMode = "select" | "create";

export interface VideoState {
  path: string;
  frameCount: number;
  fps: number;
  width: number;
  height: number;
  labeledFrames: number[];
}

interface StudioState {
  dir: string | null;
  images: ImageInfo[];
  video: VideoState | null; // non-null => video annotation mode
  currentIndex: number; // image index, or frame index in video mode
  imageSize: { width: number; height: number } | null;

  shapes: Shape[];
  flags: Record<string, unknown>;
  checked: boolean;
  otherData: Record<string, unknown>;
  hidden: Record<number, boolean>;
  selected: number | null;
  dirty: boolean;

  // undo/redo history of shapes snapshots (cleared on image switch)
  past: Shape[][];
  future: Shape[][];
  transientEdit: boolean;

  mode: EditorMode;
  createType: ShapeType;
  fitRequest: number; // bump to ask the canvas to fit the image
  samMode: boolean; // SAM point/box prompt interaction (image mode only)

  openDir: (path: string) => Promise<void>;
  openVideo: (path: string) => Promise<void>;
  closeVideo: () => Promise<void>;
  closeSession: () => void; // back to the welcome page
  selectImage: (index: number) => Promise<void>;
  saveCurrent: () => Promise<boolean>;
  nextImage: () => Promise<void>;
  prevImage: () => Promise<void>;
  copyPrevFrame: () => void;

  setMode: (mode: EditorMode) => void;
  setCreateType: (t: ShapeType) => void;
  requestFit: () => void;
  setSamMode: (on: boolean) => void;

  addShape: (s: Shape) => void;
  updateShape: (i: number, s: Shape, opts?: { history?: boolean }) => void;
  removeShape: (i: number) => void;
  beginShapeEdit: () => void; // snapshot once before a transient vertex drag
  endShapeEdit: () => void;
  undo: () => void;
  redo: () => void;
  setSelected: (i: number | null) => void;
  toggleHidden: (i: number) => void;
  setImageSize: (w: number, h: number) => void;
  setShapesExternal: (shapes: Shape[]) => void;
  markAllLabeled: () => void;
  reloadCurrent: () => Promise<void>;
  refreshImages: () => Promise<void>;
}

function normalizeShape(s: Shape): Shape {
  return {
    ...s,
    description: s.description ?? "",
    flags: s.flags ?? {},
    attributes: s.attributes ?? {},
    group_id: s.group_id ?? null,
  };
}

/** Strip the template's basic fields, keeping only extra/custom fields. */
function pickExtraFields(data: Record<string, unknown>): Record<string, unknown> {
  const {
    version,
    imagePath,
    imageData,
    imageHeight,
    imageWidth,
    shapes,
    flags,
    checked,
    ...rest
  } = data;
  return rest;
}

function applyLabelData(
  set: (partial: Partial<StudioState>) => void,
  res: { exists: boolean; data: Record<string, unknown> | null },
  extra?: Partial<StudioState>
) {
  if (res.exists && res.data) {
    const data = res.data;
    set({
      shapes: ((data.shapes as Shape[]) ?? []).map(normalizeShape),
      flags: (data.flags as Record<string, unknown>) ?? {},
      checked: (data.checked as boolean) ?? false,
      otherData: pickExtraFields(data),
      dirty: false,
      ...extra,
    });
  } else {
    set({ shapes: [], flags: {}, checked: false, otherData: {}, dirty: false, ...extra });
  }
}

export const useStudio = create<StudioState>((set, get) => {
  const pushHistory = () => {
    const { shapes, past } = get();
    set({ past: [...past.slice(-99), shapes], future: [] });
  };

  const clearHistory = { past: [] as Shape[][], future: [] as Shape[][] };

  const SESSION_KEY = "xaw_last_session";
  const saveSession = (type: "dir" | "video", path: string) => {
    try {
      localStorage.setItem(SESSION_KEY, JSON.stringify({ type, path }));
    } catch {
      /* ignore */
    }
  };

  return {
    dir: null,
    images: [],
    video: null,
    currentIndex: -1,
    imageSize: null,

    shapes: [],
    flags: {},
    checked: false,
    otherData: {},
    hidden: {},
    selected: null,
    dirty: false,

    past: [],
    future: [],
    transientEdit: false,

    mode: "select",
    createType: "rectangle",
    fitRequest: 0,
    samMode: false,

    openDir: async (path) => {
      const res = await api.openDir(path);
      set({
        dir: res.dir,
        images: res.images,
        video: null,
        currentIndex: -1,
        shapes: [],
        dirty: false,
        selected: null,
        hidden: {},
        ...clearHistory,
      });
      saveSession("dir", res.dir);
      if (res.images.length > 0) {
        await get().selectImage(0);
      }
    },

    openVideo: async (path) => {
      const info = await api.openVideo(path);
      set({
        dir: null,
        images: [],
        video: {
          path: info.video,
          frameCount: info.frame_count,
          fps: info.fps,
          width: info.width,
          height: info.height,
          labeledFrames: info.labeled_frames,
        },
        imageSize: { width: info.width, height: info.height },
        currentIndex: -1,
        shapes: [],
        dirty: false,
        selected: null,
        hidden: {},
        ...clearHistory,
      });
      saveSession("video", info.video);
      if (info.frame_count > 0) {
        await get().selectImage(0);
      }
    },

    closeVideo: async () => {
      await api.closeVideo().catch(() => undefined);
      set({
        video: null,
        currentIndex: -1,
        shapes: [],
        dirty: false,
        selected: null,
        hidden: {},
        ...clearHistory,
      });
    },

    closeSession: () => {
      const { video, dirty } = get();
      if (dirty && !window.confirm("当前有未保存的更改，关闭后将丢失。继续？")) {
        return;
      }
      if (video) {
        api.closeVideo().catch(() => undefined);
      }
      try {
        localStorage.removeItem(SESSION_KEY);
      } catch {
        /* ignore */
      }
      set({
        dir: null,
        images: [],
        video: null,
        currentIndex: -1,
        imageSize: null,
        shapes: [],
        flags: {},
        checked: false,
        otherData: {},
        hidden: {},
        selected: null,
        dirty: false,
        ...clearHistory,
      });
    },

    selectImage: async (index) => {
      const { images, video, dirty } = get();
      const total = video ? video.frameCount : images.length;
      if (index < 0 || index >= total) return;
      if (dirty && !window.confirm("当前有未保存的更改，切换后将丢失。继续？")) {
        return;
      }
      set({ currentIndex: index, selected: null, hidden: {}, dirty: false, ...clearHistory });
      try {
        if (video) {
          const res = await api.getVideoLabels(index);
          applyLabelData(set, res);
        } else {
          const res = await api.getLabels(images[index].filename);
          applyLabelData(set, res);
        }
      } catch (e) {
        console.error("load labels failed", e);
        set({ shapes: [], flags: {}, checked: false, otherData: {}, dirty: false });
      }
    },

    saveCurrent: async () => {
      const {
        images,
        video,
        currentIndex,
        shapes,
        flags,
        otherData,
        checked,
        imageSize,
      } = get();
      if (currentIndex < 0) return false;

      if (video) {
        await api.putVideoLabels(currentIndex, {
          shapes,
          flags,
          other_data: { ...otherData, checked },
          image_height: imageSize?.height ?? video.height,
          image_width: imageSize?.width ?? video.width,
        });
        set((st) => ({
          dirty: false,
          video: st.video
            ? {
              ...st.video,
              labeledFrames: st.video.labeledFrames.includes(currentIndex)
                ? st.video.labeledFrames
                : [...st.video.labeledFrames, currentIndex].sort((a, b) => a - b),
            }
            : null,
        }));
        return true;
      }

      const filename = images[currentIndex].filename;
      await api.putLabels({
        image: filename,
        shapes,
        flags,
        other_data: { ...otherData, checked },
        image_height: imageSize?.height ?? -1,
        image_width: imageSize?.width ?? -1,
      });
      set((st) => ({
        dirty: false,
        images: st.images.map((im, i) =>
          i === currentIndex ? { ...im, has_label: true } : im
        ),
      }));
      return true;
    },

    nextImage: async () => {
      const { currentIndex, images, video } = get();
      const total = video ? video.frameCount : images.length;
      if (currentIndex < total - 1) await get().selectImage(currentIndex + 1);
    },

    prevImage: async () => {
      const { currentIndex } = get();
      if (currentIndex > 0) await get().selectImage(currentIndex - 1);
    },

    copyPrevFrame: () => {
      const { currentIndex, video, shapes } = get();
      if (!video || currentIndex <= 0) return;
      // load previous frame's labels and stamp them onto the current frame
      api.getVideoLabels(currentIndex - 1).then((res) => {
        if (res.exists && res.data) {
          const prevShapes = ((res.data.shapes as Shape[]) ?? []).map(normalizeShape);
          if (prevShapes.length === 0) return;
          pushHistory();
          set({
            shapes: [...shapes, ...prevShapes],
            dirty: true,
            selected: null,
          });
        }
      });
    },

    setMode: (mode) => set({ mode, selected: null }),
    setCreateType: (t) => set({ createType: t, mode: "create", selected: null }),
    requestFit: () => set((st) => ({ fitRequest: st.fitRequest + 1 })),
    setSamMode: (on) => set({ samMode: on, selected: null }),

    addShape: (s) => {
      pushHistory();
      set((st) => ({ shapes: [...st.shapes, s], dirty: true, selected: st.shapes.length }));
    },

    updateShape: (i, s, opts) => {
      if (opts?.history !== false) pushHistory();
      set((st) => ({
        shapes: st.shapes.map((old, idx) => (idx === i ? s : old)),
        dirty: true,
      }));
    },

    removeShape: (i) => {
      pushHistory();
      set((st) => ({
        shapes: st.shapes.filter((_, idx) => idx !== i),
        selected: null,
        dirty: true,
      }));
    },

    beginShapeEdit: () => {
      if (!get().transientEdit) {
        pushHistory();
        set({ transientEdit: true });
      }
    },

    endShapeEdit: () => set({ transientEdit: false }),

    undo: () => {
      const { past, shapes, future } = get();
      if (past.length === 0) return;
      set({
        past: past.slice(0, -1),
        future: [shapes, ...future].slice(0, 100),
        shapes: past[past.length - 1],
        dirty: true,
        selected: null,
      });
    },

    redo: () => {
      const { past, shapes, future } = get();
      if (future.length === 0) return;
      const [next, ...rest] = future;
      set({
        past: [...past, shapes].slice(-100),
        future: rest,
        shapes: next,
        dirty: true,
        selected: null,
      });
    },

    setSelected: (i) => set({ selected: i }),

    toggleHidden: (i) =>
      set((st) => ({ hidden: { ...st.hidden, [i]: !st.hidden[i] } })),

    setImageSize: (w, h) => set({ imageSize: { width: w, height: h } }),

    setShapesExternal: (shapes) => {
      pushHistory();
      set({ shapes: shapes.map(normalizeShape), dirty: true, selected: null });
    },

    markAllLabeled: () =>
      set((st) => ({
        images: st.images.map((im) => ({ ...im, has_label: true })),
      })),

    reloadCurrent: async () => {
      const { currentIndex, images, video } = get();
      if (currentIndex < 0) return;
      try {
        if (video) {
          const res = await api.getVideoLabels(currentIndex);
          applyLabelData(set, res, { selected: null });
        } else {
          const res = await api.getLabels(images[currentIndex].filename);
          applyLabelData(set, res, { selected: null });
        }
      } catch (e) {
        console.error("reload labels failed", e);
      }
    },

    refreshImages: async () => {
      try {
        const { video } = get();
        if (video) {
          const info = await api.getVideoInfo();
          set((st) => ({
            video: st.video ? { ...st.video, labeledFrames: info.labeled_frames } : null,
          }));
        } else {
          const res = await api.getDirImages();
          set({ images: res.images });
        }
      } catch (e) {
        console.error("refresh images failed", e);
      }
    },
  };
});
