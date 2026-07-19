import { useEffect, useMemo, useRef, useState } from "react";
import { AutoComplete, Checkbox, Input, InputNumber, Modal } from "antd";
import { useStudio } from "../store/useStudio";

export interface LabelFormValue {
  label: string;
  group_id: number | null;
  description: string;
  difficult: boolean;
}

interface Props {
  open: boolean;
  initial?: Partial<LabelFormValue>;
  title?: string;
  onOk: (v: LabelFormValue) => void;
  onCancel: () => void;
}

export default function LabelDialog({ open, initial, title, onOk, onCancel }: Props) {
  const shapes = useStudio((s) => s.shapes);
  const [label, setLabel] = useState("");
  const [groupId, setGroupId] = useState<number | null>(null);
  const [description, setDescription] = useState("");
  const [difficult, setDifficult] = useState(false);
  const inputRef = useRef<{ focus: () => void } | null>(null);

  const labelOptions = useMemo(() => {
    const uniq = Array.from(new Set(shapes.map((s) => s.label))).filter(Boolean);
    return uniq.map((l) => ({ value: l }));
  }, [shapes]);

  useEffect(() => {
    if (open) {
      setLabel(initial?.label ?? "");
      setGroupId(initial?.group_id ?? null);
      setDescription(initial?.description ?? "");
      setDifficult(initial?.difficult ?? false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const submit = () => {
    if (!label.trim()) return;
    onOk({ label: label.trim(), group_id: groupId, description, difficult });
  };

  return (
    <Modal
      open={open}
      title={title ?? "输入标签"}
      okText="确定"
      cancelText="取消"
      onOk={submit}
      onCancel={onCancel}
      okButtonProps={{ disabled: !label.trim() }}
      destroyOnHidden
      width={380}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 8 }}>
        <div>
          <div style={{ marginBottom: 4 }}>标签名称</div>
          <AutoComplete
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            ref={inputRef as any}
            style={{ width: "100%" }}
            options={labelOptions}
            value={label}
            onChange={setLabel}
            onSelect={setLabel}
            filterOption={(input, option) =>
              (option?.value ?? "").toLowerCase().includes(input.toLowerCase())
            }
            placeholder="选择或输入新标签"
          >
            <Input onPressEnter={submit} />
          </AutoComplete>
        </div>
        <div>
          <div style={{ marginBottom: 4 }}>组 ID（可选）</div>
          <InputNumber
            style={{ width: "100%" }}
            min={0}
            value={groupId}
            onChange={(v) => setGroupId(v)}
            placeholder="相同组 ID 的形状属于同一组"
          />
        </div>
        <div>
          <div style={{ marginBottom: 4 }}>描述（可选）</div>
          <Input.TextArea
            rows={2}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
        </div>
        <Checkbox checked={difficult} onChange={(e) => setDifficult(e.target.checked)}>
          困难样本（difficult）
        </Checkbox>
      </div>
    </Modal>
  );
}
