# 远程训练教程：连接你自己的服务器

浏览器跑在本机，训练在远程服务器上执行——训练中心的「执行位置 → 远程服务器」就是为这个场景做的。本教程手把手带你从零接好一台服务器（云服务器、实验室主机、虚拟机都适用），并附常见问题的解决方案。

---

## 一、原理一分钟看懂

TrainLens 通过 **SSH** 连接你的服务器，自动完成：上传数据集 → 远端跑训练 → 日志和曲线实时回传 → 训练产物（best.pt 等）自动下载回本地。

你需要准备的只有三样：

1. 服务器的 **IP 地址** 和 SSH 端口（默认 22）
2. 一种**登录凭证**：SSH 密钥（推荐）或密码
3. 远端一个装了 `torch` + `ultralytics` 的 **Python 环境**（没有也不怕，下文教你装）

---

## 二、手把手：以阿里云 ECS 为例

### 第 1 步：拿到服务器公网 IP

控制台 → 云服务器 ECS → 实例 → 实例列表里 "IP 地址" 列，括号标着「公」的那个就是。

### 第 2 步：能登上服务器（网页终端）

新服务器先重置密码：选中实例 → 「更多」→「密码/密钥」→「重置实例密码」→ **重启实例生效**。

然后点实例右侧「远程连接」→「Workbench 远程连接」，用户名 `root` + 刚设的密码登录。看到黑色终端窗口就说明进去了。

> 有自己的 SSH 工具（Xshell / 终端）也可以，Workbench 只是零安装的选择。

### 第 3 步：配置密钥登录（推荐）或密码登录

**方式 A：SSH 密钥（推荐，更安全）**

在你**自己电脑**上生成一对密钥（Windows 用 Git Bash / PowerShell 都行）：

```bash
ssh-keygen -t ed25519 -f ~/.ssh/trainlens_server -N "" -C "trainlens"
# 查看公钥内容并完整复制:
cat ~/.ssh/trainlens_server.pub
```

然后在服务器的终端（Workbench）里执行（把整串公钥原样贴进去）：

```bash
mkdir -p ~/.ssh && echo '把你复制的公钥完整粘贴在这里（ssh-ed25519 开头）' >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys && echo OK
```

看到 `OK` 即成功。⚠️ **公钥必须完整复制**——真实案例：粘贴时丢了中间 4 个字符，连了半天都是 `Permission denied`。

**方式 B：密码登录**

什么都不用配，TrainLens 档案里选「密码认证」即可（密码只在本次会话使用，不会保存）。

### 第 4 步：安全组 / 防火墙放行 SSH 端口

云服务器都有安全组，默认可能不放行 22 端口：

- 控制台 → 安全组 → 找到实例所属安全组 →「配置规则」→ 入方向 → 增加规则
- 协议 `自定义 TCP`，端口 `22/22`，授权对象填**你本机的公网 IP**（浏览器访问 https://ifconfig.me 查看），格式如 `203.0.113.10/32`
- ⚠️ 家庭/办公宽带的出口 IP 是**动态的**，过几天变了就连不上了——到时候改一下这条规则即可；长期用建议在公司固定出口或跳板机下使用

> 服务器在局域网（比如实验室主机、VMware 虚拟机）则没有安全组，保证本机 `ping` 通即可。

### 第 5 步：在 TrainLens 里建档案并验证

训练中心 → 执行位置 → 远程服务器 → **管理服务器** → 新增档案：

| 字段 | 填什么 | 示例 |
|---|---|---|
| 名称 | 随便取 | 阿里云北京 |
| 主机 | 公网 IP | 39.x.x.x |
| 端口 | SSH 端口 | 22 |
| 用户名 | 登录用户 | root |
| 认证方式 | SSH 密钥 / 密码 | 密钥选第 3 步生成的**私钥**文件路径 |
| 远端工作目录 | 数据集/任务的存放根目录 | /root/trainlens |
| 远端 Python | 训练环境的解释器路径 | /root/trainlens-venv/bin/python |

然后：

1. **测试连接** —— 首次连接会弹出「主机指纹确认」，核对后确认（这是防中间人攻击的 TOFU 机制，指纹会记住，以后不再问；指纹变了会拒绝并告警）
2. **检测服务器** —— 自动体检远端 GPU / torch / ultralytics / 磁盘，并按结果帮你选好 CPU 或 GPU

### 第 6 步：远端训练环境（诊断缺什么装什么）

如果「检测服务器」报 PyTorch / Ultralytics 未安装，在服务器终端执行（Ubuntu/Debian 示例）：

```bash
# venv 支持和 opencv 依赖（新系统必装）
apt-get update && apt-get install -y python3-venv libgl1 libglib2.0-0

# 建一个训练专用虚拟环境
python3 -m venv /root/trainlens-venv

# 装 PyTorch（无 GPU 的服务器装 CPU 版，体积小得多）
/root/trainlens-venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# 有 NVIDIA GPU 的服务器直接: /root/trainlens-venv/bin/pip install torch torchvision

# 装 ultralytics（国内服务器用镜像飞快）
/root/trainlens-venv/bin/pip install ultralytics -i https://mirrors.aliyun.com/pypi/simple/
```

档案里的「远端 Python」就填 `/root/trainlens-venv/bin/python`。

### 第 7 步：开始训练

训练中心表单选好任务/模型/数据集，执行位置选远程服务器，「检查并启动」。日志、曲线、ETA 和本地训练完全一样；训练结束后产物自动回传，历史记录里统一下载/导出/注册。

---

## 三、常见问题（FAQ）

**Q：连接超时 / 一直转圈**
- 安全组没放行 22 端口，或放行规则里的 IP 不是你的当前出口 IP（宽带 IP 会变，用 ifconfig.me 重查）
- 规则加错了安全组：实例详情页确认它属于哪个安全组
- 服务器没开机 / sshd 没运行：`systemctl status ssh`

**Q：Permission denied (publickey,password)**
- 公钥没写进 `~/.ssh/authorized_keys`，或**复制不完整**（最常见！对照本机 `cat ~/.ssh/xxx.pub` 的输出逐字符检查）
- 公钥写给了错误的用户：`authorized_keys` 是「按用户」的，Workbench 用谁登录就写给谁
- 权限不对：`.ssh` 必须 700、`authorized_keys` 必须 600（第 3 步命令已包含）

**Q：提示「主机密钥指纹与已保存的不一致」**
- 服务器重装系统 / 重建实例后指纹会变，属正常；确认是自己在操作后，删除档案重新建即可重置指纹
- 如果你没重装过，请警惕中间人攻击，先别连

**Q：诊断报 "PyTorch is not installed" / "Ultralytics is not installed"**
- 按第 6 步装环境；注意装到了哪个 venv，档案的「远端 Python」必须指向**那个 venv 里的 python**，不是系统 python

**Q：`python3 -m venv` 报错 / pip 不存在**
- Ubuntu/Debian 先 `apt-get install -y python3-venv`（新版系统 venv 是独立包）

**Q：训练时 opencv 报 libGL 之类的错**
- `apt-get install -y libgl1 libglib2.0-0`（无桌面系统的常见缺库）

**Q：pip 安装特别慢**
- 国内服务器加镜像参数：`-i https://mirrors.aliyun.com/pypi/simple/`
- 无 GPU 的服务器务必装 CPU 版 torch（体积小几倍）：`--index-url https://mirrors.aliyun.com/pytorch-wheels/cpu`
- 低配 ECS 出口带宽小（如 1-2Mbps）时，再大镜像也快不了：在本机（网络好的话）`pip download torch torchvision ultralytics --only-binary=:all: --python-version 312 --platform manylinux_2_28_x86_64 -d wheels`，`scp -r wheels/ root@服务器:/root/`，然后服务器上 `pip install --no-index --find-links=/root/wheels torch torchvision ultralytics`

**Q：启动训练后等了很久才看到日志**
- 正常现象：首次训练前要把整个数据集打包上传到服务器，数据集越大越久；日志里的上传进度阶段请耐心等待（中断用「停止」即可，续传版本在规划中）

**Q：服务器内存小（如 2G）训练会炸吗**
- 把训练参数调小：batch 4 甚至 2、imgsz 320/416，CPU 上小数据验证流程完全够用；正式训练建议 4G 以上内存

**Q：Workbench 和 TrainLens 远程训练是什么关系？**
- Workbench 只是阿里云提供的网页版终端，用于你对服务器做初始化（写公钥、装环境）；TrainLens 的远程训练是它自己通过 SSH 直连服务器完成的，之后不再需要 Workbench

---

## 四、安全建议

- 优先用**密钥认证**；密码认证只在内网/临时场景用，密码不落盘（每次启动任务时填写）
- 安全组授权对象尽量写具体 IP（`x.x.x.x/32`），不要长期挂 `0.0.0.0/0`
- 主机指纹变了先核实再重建档案
- 用完可以删规则/关实例，TrainLens 的档案保存在本机，下次直接测连接即可
