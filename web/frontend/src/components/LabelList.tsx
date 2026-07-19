import { Button, Empty, List, Popconfirm, Tag, Tooltip } from "antd";
import {
  DeleteOutlined,
  EditOutlined,
  EyeInvisibleOutlined,
  EyeOutlined,
} from "@ant-design/icons";
import { useStudio } from "../store/useStudio";
import { labelColor } from "../utils/colors";

const TYPE_NAME: Record<string, string> = {
  rectangle: "矩形",
  polygon: "多边形",
  rotation: "旋转框",
  circle: "圆",
  line: "线",
  point: "点",
  linestrip: "折线",
  cuboid: "立方体",
};

interface Props {
  onEditLabel: (index: number) => void;
}

export default function LabelList({ onEditLabel }: Props) {
  const { shapes, selected, hidden, setSelected, toggleHidden, removeShape } =
    useStudio();

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div
        style={{
          padding: "8px 12px",
          borderBottom: "1px solid #f0f0f0",
          fontWeight: 600,
        }}
      >
        标签（{shapes.length}）
      </div>
      <div style={{ flex: 1, overflow: "auto" }}>
        {shapes.length === 0 ? (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description="暂无标注"
            style={{ marginTop: 40 }}
          />
        ) : (
          <List
            size="small"
            dataSource={shapes.map((s, i) => ({ s, i }))}
            renderItem={({ s, i }) => (
              <List.Item
                onClick={() => setSelected(i)}
                style={{
                  cursor: "pointer",
                  padding: "4px 8px",
                  background: selected === i ? "#e6f4ff" : undefined,
                  opacity: hidden[i] ? 0.4 : 1,
                }}
                actions={[
                  <Tooltip title={hidden[i] ? "显示" : "隐藏"} key="v">
                    <Button
                      type="text"
                      size="small"
                      icon={hidden[i] ? <EyeInvisibleOutlined /> : <EyeOutlined />}
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleHidden(i);
                      }}
                    />
                  </Tooltip>,
                  <Tooltip title="编辑标签" key="e">
                    <Button
                      type="text"
                      size="small"
                      icon={<EditOutlined />}
                      onClick={(e) => {
                        e.stopPropagation();
                        onEditLabel(i);
                      }}
                    />
                  </Tooltip>,
                  <Popconfirm
                    key="d"
                    title="删除该标注？"
                    onConfirm={() => removeShape(i)}
                  >
                    <Button
                      type="text"
                      size="small"
                      danger
                      icon={<DeleteOutlined />}
                      onClick={(e) => e.stopPropagation()}
                    />
                  </Popconfirm>,
                ]}
              >
                <span
                  style={{
                    display: "inline-block",
                    width: 10,
                    height: 10,
                    borderRadius: 2,
                    background: labelColor(s.label),
                    marginRight: 8,
                    flexShrink: 0,
                  }}
                />
                <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis" }}>
                  {s.label}
                  {s.group_id != null && (
                    <Tag style={{ marginLeft: 6 }} color="blue">
                      G{s.group_id}
                    </Tag>
                  )}
                </span>
                <span style={{ color: "#bbb", fontSize: 11, marginLeft: 4 }}>
                  {TYPE_NAME[s.shape_type] ?? s.shape_type}
                </span>
              </List.Item>
            )}
          />
        )}
      </div>
    </div>
  );
}
