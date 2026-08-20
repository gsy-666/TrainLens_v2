import { useEffect, useMemo, useState } from "react";
import { Tour } from "antd";
import type { TourProps } from "antd";

export interface GuidedTourStep {
  /** 目标元素的 DOM id；元素不存在时该步骤自动跳过 */
  targetId: string;
  title: string;
  description: string;
}

/**
 * 页面新手引导的触发逻辑：首次进入自动弹出一次（localStorage 标记），
 * 之后可通过返回的 openTour 手动重看。
 */
export function useGuidedTour(storageKey: string) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (localStorage.getItem(storageKey)) return;
    // 延迟弹出，等待页面 DOM 渲染完成
    const timer = window.setTimeout(() => {
      localStorage.setItem(storageKey, "1");
      setOpen(true);
    }, 500);
    return () => window.clearTimeout(timer);
  }, [storageKey]);

  return {
    open,
    openTour: () => setOpen(true),
    closeTour: () => setOpen(false),
  };
}

interface Props {
  steps: GuidedTourStep[];
  open: boolean;
  onClose: () => void;
}

export default function GuidedTour({ steps, open, onClose }: Props) {
  // 打开时才解析目标元素并裁剪掉不存在的步骤（如视频模式下缺失的按钮）
  const resolvedSteps = useMemo<TourProps["steps"]>(() => {
    if (!open) return [];
    return steps
      .filter((s) => document.getElementById(s.targetId) !== null)
      .map((s) => ({
        title: s.title,
        description: s.description,
        target: () => document.getElementById(s.targetId) as HTMLElement,
      }));
  }, [open, steps]);

  return <Tour open={open} onClose={onClose} steps={resolvedSteps} />;
}
