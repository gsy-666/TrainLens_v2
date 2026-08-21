# TrainLens Web UI

<p>
  <img src="frontend/public/logo.png" alt="TrainLens" width="200" />
</p>

桌面版 X-AnyLabeling(https://github.com/CVHub520/X-AnyLabeling) 的 Web 实现：**React 前端 + FastAPI 后端**，标注 JSON 与桌面版逐字段兼容，AI 推理、格式导出、训练管线直接复用桌面版代码，两个版本可以打开同一份数据集无缝协作。

![标注效果](frontend/public/annotation-preview.jpg)

## 功能

### 标注
- 8 种形状：矩形 / 多边形 / 旋转框 / 圆 / 直线 / 点 / 折线 / 立方体（cuboid）
- 顶点拖拽编辑、整体移动、显隐、复制、删除，标签自动补全、group_id、描述、difficult
- 图像目录标注 + **视频逐帧标注**（帧滑杆 / 跳转 / 复制上一帧）
- 快捷键：`A/D` 切换、`Ctrl+S` 保存、`V/R/P/O/C/L/T/S/U` 切换工具、`Del` 删除、`Esc` 取消、`F` 适应窗口

### AI 自动标注
- **190+ 内置模型**（YOLO 系列 / SAM / Grounding-DINO / PPOCR 等），选中即自动下载（带进度条），已下载自动跳过
- **模型库扫描**：下拉直接标记「已下载 / 自定义」，模型库弹窗查看本机已有模型与磁盘占用、删除缓存释放空间、扫描任意目录发现散落 .onnx 一键注册
- **本地权重注册**：本机已有的 .onnx 可通过「使用本地权重文件」选个模板模型直接注册加载，无需下载
- 单图推理、批量预标注、置信度 / IoU 阈值调节、文本提示（Grounding 类模型）
- **视频 MOT 跟踪**（bytetrack / botsort / tracktrack 类模型），跨帧一致的跟踪 ID（group_id）
- **一键撤回**：批量预标注和跟踪覆盖写入的标注都可以一键恢复；之后被手动修改的文件自动跳过

### 数据
- 本地目录浏览器（盘符 / 文件夹 / 文件，含图片目录标识）
- 上传图片 / 标注文件到当前数据集
- 导出：**YOLO**（hbb/obb/seg）、**Pascal VOC**（检测/分割）、**COCO**（检测/分割）、**DOTA**、**Mask**、**MOT**、**ODVG**，类别自动提取，支持 ZIP 打包下载

### 训练与监控
- **标注 → 训练闭环**：训练中心可一键把当前标注目录（Labelme JSON）转换为 YOLO 训练集（自动提取类别、分层抽样划分 train/val、生成 data.yaml），并自动填入训练表单，直接开训
- **数据集检查可视**：类别×形状统计、类别实例分布图、5 种任务有效图片数（自动推荐任务类型）、少样本/类别不均衡/数据量不足告警
- **训练中心**：Ultralytics 引导式训练，环境/数据集预检查（`POST /api/training/preflight`）、新手三档预设、实时日志、loss / mAP / 学习率曲线、ETA 预计剩余时间、历史记录（与桌面版共享存储）
- **远程服务器训练**：SSH 服务器档案管理（密钥/密码认证、host-key 确认）、远端环境诊断（GPU / torch / ultralytics / 磁盘）、自动选择 CPU/GPU、数据集自动上传、训练产物自动回传，日志曲线与本地任务同界面
- **产物转化**：best.pt / last.pt 在线预览曲线图、下载 / ZIP 打包、导出 ONNX 等格式（按环境动态检测可用性）、**一键注册为标注模型**——训练完直接在标注页加载试用，验证效果后继续标注
- **模型试用 Playground**：训练中心内置试玩面板，选模型→传图→即时查看检测框与置信度，无需打开数据集
- **难例优先（主动学习）**：用已加载模型扫描未标注图片并按置信度打难度分，文件列表按"最没把握"排序，人优先复核难例，越标越准
- **新手引导**：标注页与训练中心内置分步高亮 Tour，首次进入自动弹出，可随时点"?"重看
- **运行监控**：自定义脚本工作区扫描（脚本/环境检测）、启动/停止、stdout/stderr 实时流、进程 CPU/内存 + 系统 + GPU 资源曲线

## 快速开始（一键启动）

**Windows**：双击 `start_web.bat`
**Linux / macOS**：`bash start_web.sh`

脚本自动完成：检查后端依赖 → 首次构建前端 → 启动服务 → 打开浏览器。

- 访问 **http://127.0.0.1:8000**（单进程，API 与页面同源）
- 前提：项目 Python 环境（仓库 `.venv` 或已安装项目依赖的 Python）；首次构建前端需 Node.js
- `Ctrl+C` 停止
- 再次启动会自动恢复上次打开的图片/视频目录（会话恢复）

## 远程训练（SSH 下发到 GPU 服务器）

浏览器跑在本机、训练在远程服务器上执行的场景（与下面"远程访问"是两个独立功能，可叠加）：

1. 训练中心 → 执行位置 → 远程服务器 → 管理服务器，新增档案：主机 / 端口 / 用户名 / 认证方式（SSH 密钥或密码）/ 远端工作目录 / 远端 Python 路径（需已安装 torch + ultralytics 的环境）
2. 首次连接按提示确认 host-key 指纹；"检测服务器"自动体检远端 GPU、torch、ultralytics 版本并按结果自动选择 CPU/GPU
3. 启动训练后，数据集自动打包上传、远端执行训练、实时日志与曲线回流、训练结束产物（best.pt / last.pt / results.csv / 曲线图）自动回传本地，历史记录与本地任务统一展示

## 远程访问（云服务器部署）

训练在云服务器上跑的场景：把仓库放到服务器，一键启动时加 `--host` 参数：

```bash
bash start_web.sh --host 0.0.0.0            # 或 start_web.bat --host 0.0.0.0
```

- 启动器会**自动生成访问令牌**并打印在控制台（`--token XXX` 可指定固定令牌，或用环境变量 `XANYLABELING_WEB_TOKEN`）
- 模型下载走代理：加 `--proxy http://127.0.0.1:7890`（下载慢时也可手动把 onnx 放到 `~/anylabeling_data/models/<模型名>/`，加载器自动跳过下载；也可用模型面板的「使用本地权重文件」直接注册本机已有的 .onnx）
- 模型缓存目录自定义：设置环境变量 `XANYLABELING_MODELS_DIR`（例如 D 盘）后，自动下载的模型改为存到该目录
- 本地浏览器打开 `http://<服务器IP>:8000`，页面会要求输入令牌，输入即可使用
- 令牌对全部 API 生效（含图片流和文件下载），本地回环访问免令牌
- **安全提示**：HTTP 下令牌为明文传输，请只在可信网络/内网使用；暴露在公网时建议改用 SSH 隧道（零改动、全程加密）：
  `ssh -L 8000:127.0.0.1:8000 user@server`，然后本地访问 http://127.0.0.1:8000

## 开发模式

**Windows**：双击 `start_dev.bat`（后端 `--reload` + Vite 热更新，访问 http://localhost:5173）

或手动：

```bash
# 后端
cd web/backend && ../../.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
# 前端
cd web/frontend && npm install && npm run dev
```

前端改动后重新 `npm run build`，一键模式即用上新版。

## 架构

```
web/
├── start_web.bat / start_web.sh   # 一键启动（生产模式，单进程 :8000）
├── start_dev.bat                  # 开发模式（:8000 + :5173）
├── backend/app/
│   ├── main.py                    # FastAPI 入口，CORS，托管 frontend/dist
│   ├── routers/                   # files / labels / models / predict / export / upload / video / training / dataset / quickstart / monitor / fs / system / remote
│   ├── model_service.py           # 复用桌面版 ModelManager（ONNX 推理）
│   ├── training_service.py        # 复用 training_center（JobManager / MetricStore / HistoryStore / Preflight）
│   ├── web_runner.py              # 去 Qt 的本地训练 runner（subprocess + threading）
│   ├── web_ssh_runner.py          # 去 Qt 的 SSH 远程训练 runner（paramiko 流式读取）
│   ├── run_service.py             # 纯 threading 进程管理 + psutil 资源采样
│   ├── backup.py                  # 自动标注覆盖写入的一级撤回
│   └── adapters.py                # 标注 JSON ⇄ 桌面 LabelFile 格式
└── frontend/src/
    ├── pages/                     # Welcome / LabelStudio / TrainingCenter / RunMonitor
    ├── components/                # CanvasEditor(react-konva) / Toolbar / FileList / LabelList / ModelPanel / ...
    └── store/useStudio.ts         # Zustand 全局状态
```

**复用的桌面版模块**：`label_file`/`schema`（标注读写）、`services/auto_labeling`（190+ ONNX 模型）、`label_converter`（7 种格式导出）、`services/training_center`（训练任务/指标/历史/预检查）、`services/run_monitor`（工作区扫描/脚本检测）。

## API 摘要

| 端点 | 说明 |
|---|---|
| `POST /api/dir/open` · `GET /api/image` | 打开图片目录 / 取图 |
| `GET/PUT/DELETE /api/labels` | 标注读写（桌面 Labelme 格式） |
| `GET /api/fs/list` | 本地目录浏览 |
| `GET/POST /api/models/*` · `POST /api/predict[/batch]` | 模型加载 / 推理 |
| `POST /api/models/register-local` · `DELETE /api/models/custom` | 注册 / 移除本地 .onnx 权重 |
| `GET /api/models/local-files` · `DELETE /api/models/cache` · `GET /api/models/scan-dir` | 模型库扫描 / 缓存清理 / 目录发现 |
| `POST /api/predict/batch/undo` | 撤回批量标注 |
| `POST /api/video/open` · `GET /api/video/frame` · `PUT /api/video/labels` · `POST /api/video/track` | 视频标注与 MOT 跟踪 |
| `POST /api/export` · `GET /api/export/download` | 数据集导出 / ZIP 下载 |
| `POST /api/dataset/prepare` · `GET /api/dataset/stats` | 训练集生成 / 数据集检查统计 |
| `POST /api/training/quickstart` | 零配置一键训练（自动推断任务/设备/模型） |
| `POST /api/training/preflight` · `POST /api/training/guided/start` · `GET /api/training/{events,metrics,history,status}` | 预检查 / 训练任务 |
| `GET /api/training/history/{id}/artifacts` · `POST .../artifacts/export` · `POST .../artifacts/register-model` | 训练产物查看 / 导出 / 注册为标注模型 |
| `GET/POST/PUT/DELETE /api/remote/profiles` · `POST .../test` · `POST .../diagnostics` | SSH 远程服务器档案 / 连接测试 / 环境诊断 |
| `POST /api/playground/predict` | 模型试用（上传图片即时推理） |
| `POST /api/active_learning/scan` · `GET .../scores` | 难例扫描 / 难度分查询 |
| `POST /api/monitor/{scan,start,stop}` · `GET /api/monitor/{logs,resources}` | 运行监控 |

完整交互式文档：启动后访问 http://127.0.0.1:8000/docs

## 兼容性

- 标注文件格式与桌面版**逐字段一致**（含矩形四点、rotation direction、cuboid3d 元数据），双向可互开
- 训练历史与桌面版共享同一存储目录
- 桌面版代码零改动，Web 版全部位于 `web/` 目录
