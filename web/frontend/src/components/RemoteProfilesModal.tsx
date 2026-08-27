import { useCallback, useEffect, useState } from "react";
import {
  Button,
  Collapse,
  Input,
  InputNumber,
  List,
  message,
  Modal,
  Popconfirm,
  Radio,
  Space,
  Tag,
  Typography,
} from "antd";
import {
  ApiOutlined,
  DeleteOutlined,
  EditOutlined,
  PlusOutlined,
  QuestionCircleOutlined,
} from "@ant-design/icons";
import * as api from "../api/client";

interface Props {
  open: boolean;
  onClose: () => void;
  onProfilesChanged?: (profiles: api.RemoteProfile[]) => void;
}

interface FormState {
  name: string;
  host: string;
  port: number;
  username: string;
  auth_method: "ssh_key" | "password";
  private_key_path: string;
  remote_workspace: string;
  remote_python: string;
  proxy: string;
}

const EMPTY_FORM: FormState = {
  name: "",
  host: "",
  port: 22,
  username: "",
  auth_method: "ssh_key",
  private_key_path: "",
  remote_workspace: "/data/trainlens",
  remote_python: "python3",
  proxy: "",
};

export default function RemoteProfilesModal({ open, onClose, onProfilesChanged }: Props) {
  const [profiles, setProfiles] = useState<api.RemoteProfile[]>([]);
  const [loading, setLoading] = useState(false);
  // null = 列表视图；"new" 或 profile_id = 编辑视图
  const [editing, setEditing] = useState<string | null>(null);
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [saving, setSaving] = useState(false);
  const [testingId, setTestingId] = useState<string | null>(null);
  const [formPassword, setFormPassword] = useState("");
  const [pwdPromptFor, setPwdPromptFor] = useState<api.RemoteProfile | null>(null);
  const [pwdInput, setPwdInput] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const d = await api.listRemoteProfiles();
      setProfiles(d.profiles);
      onProfilesChanged?.(d.profiles);
    } catch {
      message.error("加载服务器档案失败");
    } finally {
      setLoading(false);
    }
  }, [onProfilesChanged]);

  useEffect(() => {
    if (open) {
      setEditing(null);
      void load();
    }
  }, [open, load]);

  const runTest = useCallback(
    async (profile: api.RemoteProfile, password?: string) => {
      setTestingId(profile.profile_id);
      try {
        const r = await api.testRemoteProfile(profile.profile_id, password);
        if (r.ok) {
          message.success(`连接成功，主机指纹 ${r.fingerprint}`);
          setTestingId(null);
        } else if (r.need_host_key_confirm && r.fingerprint) {
          // Modal.confirm 是非阻塞的：确认框打开期间保持 testingId（测试按钮
          // 维持 loading/禁用），由 onOk / onCancel 负责复位，避免连点堆出
          // 多个确认框
          Modal.confirm({
            title: "首次连接：确认主机指纹",
            width: 480,
            content: (
              <div style={{ marginTop: 8 }}>
                <div>请确认服务器 {profile.host} 的 SSH 主机指纹：</div>
                <div
                  style={{
                    fontFamily: "Consolas, monospace",
                    fontSize: 12,
                    wordBreak: "break-all",
                    background: "#f5f5f5",
                    padding: 8,
                    borderRadius: 4,
                    margin: "8px 0",
                  }}
                >
                  {r.fingerprint}
                </div>
                <div style={{ color: "#999", fontSize: 12 }}>
                  信任后指纹将保存到档案中，之后连接自动校验；如指纹日后发生变化会被拒绝连接。
                </div>
              </div>
            ),
            okText: "信任并保存",
            cancelText: "取消",
            onOk: async () => {
              try {
                await api.confirmRemoteHostKey(profile.profile_id, r.fingerprint!);
              } catch (e) {
                const err = e as { response?: { data?: { detail?: string } }; message: string };
                message.error(`保存主机指纹失败: ${err.response?.data?.detail ?? err.message}`, 6);
                setTestingId(null);
                return;
              }
              message.success("指纹已保存，正在重新测试…");
              void load();
              // 重测结束（或再次需要确认）时由 runTest 自己复位 testingId
              await runTest(profile, password);
            },
            onCancel: () => setTestingId(null),
          });
        } else {
          message.error(`连接失败: ${r.error ?? "未知错误"}`, 6);
          setTestingId(null);
        }
      } catch (e) {
        const err = e as { response?: { data?: { detail?: string } }; message: string };
        message.error(`连接测试失败: ${err.response?.data?.detail ?? err.message}`, 6);
        setTestingId(null);
      }
    },
    [load]
  );

  const onTestClick = useCallback(
    (profile: api.RemoteProfile) => {
      if (profile.auth_method === "password") {
        setPwdInput("");
        setPwdPromptFor(profile);
      } else {
        void runTest(profile);
      }
    },
    [runTest]
  );

  const onEdit = useCallback((p: api.RemoteProfile) => {
    setForm({
      name: p.name,
      host: p.host,
      port: p.port,
      username: p.username,
      auth_method: p.auth_method,
      private_key_path: p.private_key_path,
      remote_workspace: p.remote_workspace,
      remote_python: p.remote_python,
      proxy: p.proxy ?? "",
    });
    setFormPassword("");
    setEditing(p.profile_id);
  }, []);

  const onSave = useCallback(async () => {
    setSaving(true);
    try {
      const payload: api.RemoteProfilePayload = { ...form };
      if (editing === "new") {
        await api.createRemoteProfile(payload);
        message.success("服务器档案已创建");
      } else if (editing) {
        await api.updateRemoteProfile(editing, payload);
        message.success("服务器档案已保存");
      }
      setEditing(null);
      await load();
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message: string };
      message.error(`保存失败: ${err.response?.data?.detail ?? err.message}`);
    } finally {
      setSaving(false);
    }
  }, [editing, form, load]);

  const onDelete = useCallback(
    async (profileId: string) => {
      try {
        await api.deleteRemoteProfile(profileId);
        message.success("已删除");
        await load();
      } catch (e) {
        const err = e as { response?: { data?: { detail?: string } }; message: string };
        message.error(`删除失败: ${err.response?.data?.detail ?? err.message}`);
      }
    },
    [load]
  );

  const editingProfile = editing && editing !== "new"
    ? profiles.find((p) => p.profile_id === editing) ?? null
    : null;

  return (
    <Modal
      open={open}
      title="远程服务器档案"
      onCancel={onClose}
      footer={null}
      width={640}
      destroyOnHidden
    >
      {editing === null ? (
        <>
          <Collapse
            size="small"
            style={{ marginBottom: 12 }}
            items={[
              {
                key: "guide",
                label: (
                  <span style={{ fontSize: 12 }}>
                    <QuestionCircleOutlined /> 第一次连接服务器？查看三步指引
                  </span>
                ),
                children: (
                  <div style={{ fontSize: 12, color: "#52525b", lineHeight: 1.9 }}>
                    <div>
                      <b>① 打通网络</b>：服务器开启 sshd；云服务器还需在安全组放行 22
                      端口到你的本机出口 IP（浏览器打开 ifconfig.me 查看；家用宽带 IP
                      会变化，变了改一下规则）。
                    </div>
                    <div>
                      <b>② 配置登录</b>：密钥（推荐）——本机执行{" "}
                      <Typography.Text code copyable style={{ fontSize: 11 }}>
                        ssh-keygen -t ed25519 -f ~/.ssh/trainlens_server -N ""
                      </Typography.Text>{" "}
                      生成密钥，再把生成的 .pub 公钥<b>完整</b>追加到服务器的
                      ~/.ssh/authorized_keys（目录权限 700、文件 600）；下方「私钥文件」填
                      ~/.ssh/trainlens_server。或选密码认证（密码仅本次会话使用，不保存）。
                    </div>
                    <div>
                      <b>③ 远端环境</b>：Python 里装好 torch + ultralytics（建议 venv，无 GPU
                      装 CPU 版即可），下方「远程 Python」填该环境的解释器路径。
                    </div>
                    <div style={{ color: "#71717a" }}>
                      填好点「测试连接」：首次会要求确认主机指纹（防中间人）；「检测服务器」会自动识别远端
                      CPU/GPU。完整图文教程和常见问题见仓库 web/REMOTE_TRAINING.md。
                    </div>
                  </div>
                ),
              },
            ]}
          />
          <div style={{ marginBottom: 12, display: "flex", justifyContent: "space-between" }}>
            <span style={{ fontSize: 12, color: "#999" }}>
              密码认证的登录密码永不保存，仅在测试连接 / 启动训练时填写。
            </span>
            <Button
              type="primary"
              size="small"
              icon={<PlusOutlined />}
              onClick={() => {
                setForm(EMPTY_FORM);
                setFormPassword("");
                setEditing("new");
              }}
            >
              新增服务器
            </Button>
          </div>
          <List
            size="small"
            loading={loading}
            dataSource={profiles}
            locale={{ emptyText: "暂无服务器档案" }}
            renderItem={(p) => (
              <List.Item
                actions={[
                  <Button
                    key="test"
                    size="small"
                    icon={<ApiOutlined />}
                    loading={testingId === p.profile_id}
                    disabled={testingId !== null}
                    onClick={() => onTestClick(p)}
                  >
                    测试连接
                  </Button>,
                  <Button key="edit" size="small" icon={<EditOutlined />} onClick={() => onEdit(p)}>
                    编辑
                  </Button>,
                  <Popconfirm
                    key="del"
                    title={`删除档案「${p.name}」？`}
                    okText="删除"
                    cancelText="取消"
                    onConfirm={() => onDelete(p.profile_id)}
                  >
                    <Button size="small" danger icon={<DeleteOutlined />} />
                  </Popconfirm>,
                ]}
              >
                <div>
                  <Space size={6}>
                    <strong>{p.name}</strong>
                    <Tag color={p.auth_method === "password" ? "orange" : "blue"}>
                      {p.auth_method === "password" ? "密码" : "密钥"}
                    </Tag>
                    {p.known_host_fingerprint && <Tag color="green">已信任</Tag>}
                  </Space>
                  <div style={{ fontSize: 12, color: "#999" }}>
                    {p.username}@{p.host}:{p.port} · {p.remote_workspace}
                  </div>
                </div>
              </List.Item>
            )}
          />
        </>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ display: "flex", gap: 8 }}>
            <div style={{ flex: 1 }}>
              <div style={{ marginBottom: 4 }}>名称</div>
              <Input
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                placeholder="例如：实验室 GPU 服务器"
              />
            </div>
            <div style={{ flex: 2 }}>
              <div style={{ marginBottom: 4 }}>主机地址</div>
              <Input
                value={form.host}
                onChange={(e) => setForm({ ...form, host: e.target.value })}
                placeholder="192.168.1.10 或 gpu.example.com"
              />
            </div>
            <div style={{ width: 90 }}>
              <div style={{ marginBottom: 4 }}>端口</div>
              <InputNumber
                style={{ width: "100%" }}
                min={1}
                max={65535}
                value={form.port}
                onChange={(v) => setForm({ ...form, port: v ?? 22 })}
              />
            </div>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <div style={{ flex: 1 }}>
              <div style={{ marginBottom: 4 }}>用户名</div>
              <Input
                value={form.username}
                onChange={(e) => setForm({ ...form, username: e.target.value })}
              />
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ marginBottom: 4 }}>认证方式</div>
              <Radio.Group
                value={form.auth_method}
                onChange={(e) =>
                  setForm({ ...form, auth_method: e.target.value as FormState["auth_method"] })
                }
                options={[
                  { value: "ssh_key", label: "SSH 密钥" },
                  { value: "password", label: "密码" },
                ]}
              />
            </div>
          </div>
          {form.auth_method === "ssh_key" ? (
            <div>
              <div style={{ marginBottom: 4 }}>私钥路径（本机文件）</div>
              <Input
                value={form.private_key_path}
                onChange={(e) => setForm({ ...form, private_key_path: e.target.value })}
                placeholder="C:/Users/you/.ssh/id_rsa"
              />
            </div>
          ) : (
            <div>
              <div style={{ marginBottom: 4 }}>登录密码（仅本次会话，不会保存）</div>
              <Input.Password
                value={formPassword}
                onChange={(e) => setFormPassword(e.target.value)}
                placeholder="测试连接 / 启动训练时使用"
              />
            </div>
          )}
          <div>
            <div style={{ marginBottom: 4 }}>远程工作目录（数据集与训练任务将上传到这里）</div>
            <Input
              value={form.remote_workspace}
              onChange={(e) => setForm({ ...form, remote_workspace: e.target.value })}
              placeholder="/data/trainlens"
            />
          </div>
          <div>
            <div style={{ marginBottom: 4 }}>远程 Python 路径（需已安装 torch / ultralytics）</div>
            <Input
              value={form.remote_python}
              onChange={(e) => setForm({ ...form, remote_python: e.target.value })}
              placeholder="python3 或 /opt/conda/envs/train/bin/python"
            />
          </div>
          <div>
            <div style={{ marginBottom: 4 }}>下载代理（可选，远端下载预训练权重时生效）</div>
            <Input
              value={form.proxy}
              onChange={(e) => setForm({ ...form, proxy: e.target.value })}
              placeholder="如 http://127.0.0.1:7890，留空表示不走代理"
            />
          </div>
          <Space style={{ marginTop: 4 }}>
            <Button type="primary" loading={saving} onClick={onSave}>
              保存
            </Button>
            {editingProfile && (
              <Button
                icon={<ApiOutlined />}
                loading={testingId === editingProfile.profile_id}
                disabled={testingId !== null}
                onClick={() => void runTest(editingProfile, formPassword || undefined)}
              >
                测试连接
              </Button>
            )}
            <Button onClick={() => setEditing(null)}>返回列表</Button>
          </Space>
        </div>
      )}

      {/* 密码认证的服务器：列表里点「测试连接」时先询问密码 */}
      <Modal
        open={!!pwdPromptFor}
        title={`输入 ${pwdPromptFor?.name ?? "服务器"} 的登录密码`}
        okText="测试连接"
        cancelText="取消"
        onCancel={() => setPwdPromptFor(null)}
        onOk={() => {
          const p = pwdPromptFor;
          setPwdPromptFor(null);
          if (p) void runTest(p, pwdInput || undefined);
        }}
        width={380}
      >
        <div style={{ marginTop: 8 }}>
          <Input.Password
            value={pwdInput}
            onChange={(e) => setPwdInput(e.target.value)}
            placeholder="仅本次会话使用，不会保存"
          />
        </div>
      </Modal>
    </Modal>
  );
}
