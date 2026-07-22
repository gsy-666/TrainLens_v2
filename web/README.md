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
- **190+ 内置模型**（YOLO 系列 / SAM / Grounding-DINO / PPOCR 等），模型自动下载并显示进度
- 单图推理、批量预标注、置信度 / IoU 阈值调节、文本提示（Grounding 类模型）
- **视频 MOT 跟踪**（bytetrack / botsort / tracktrack 类模型），跨帧一致的跟踪 ID（group_id）
- **一键撤回**：批量预标注和跟踪覆盖写入的标注都可以一键恢复；之后被手动修改的文件自动跳过

### 数据
- 本地目录浏览器（盘符 / 文件夹 / 文件，含图片目录标识）
- 上传图片 / 标注文件到当前数据集
- 导出：**YOLO**（hbb/obb/seg）、**Pascal VOC**（检测/分割）、**COCO**（检测/分割）、**DOTA**、**Mask**、**MOT**、**ODVG**，类别自动提取，支持 ZIP 打包下载

### 训练与监控
- **标注 → 训练闭环**：训练中心可一键把当前标注目录（Labelme JSON）转换为 YOLO 训练集（自动提取类别、分层抽样划分 train/val、生成 data.yaml），并自动填入训练表单，直接开训
- **训练中心**：Ultralytics 引导式训练，环境/数据集预检查、实时日志、loss / mAP / 学习率曲线、历史记录（与桌面版共享存储）
- **运行监控**：自定义脚本工作区扫描（脚本/环境检测）、启动/停止、stdout/stderr 实时流、进程 CPU/内存 + 系统 + GPU 资源曲线

## 快速开始（一键启动）

**Windows**：双击 `start_web.bat`
**Linux / macOS**：`bash start_web.sh`

脚本自动完成：检查后端依赖 → 首次构建前端 → 启动服务 → 打开浏览器。

- 访问 **http://127.0.0.1:8000**（单进程，API 与页面同源）
- 前提：项目 Python 环境（仓库 `.venv` 或已安装项目依赖的 Python）；首次构建前端需 Node.js
- `Ctrl+C` 停止

## 远程访问（云服务器部署）

训练在云服务器上跑的场景：把仓库放到服务器，一键启动时加 `--host` 参数：

```bash
bash start_web.sh --host 0.0.0.0            # 或 start_web.bat --host 0.0.0.0
```

- 启动器会**自动生成访问令牌**并打印在控制台（`--token XXX` 可指定固定令牌，或用环境变量 `XANYLABELING_WEB_TOKEN`）
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
│   ├── routers/                   # files / labels / models / predict / export / upload / video / training / monitor / fs
│   ├── model_service.py           # 复用桌面版 ModelManager（ONNX 推理）
│   ├── training_service.py        # 复用 training_center（JobManager / MetricStore / HistoryStore）
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
| `POST /api/predict/batch/undo` | 撤回批量标注 |
| `POST /api/video/open` · `GET /api/video/frame` · `PUT /api/video/labels` · `POST /api/video/track` | 视频标注与 MOT 跟踪 |
| `POST /api/export` · `GET /api/export/download` | 数据集导出 / ZIP 下载 |
| `POST /api/training/guided/start` · `GET /api/training/{events,metrics,history}` | 训练任务 |
| `POST /api/monitor/{scan,start,stop}` · `GET /api/monitor/{logs,resources}` | 运行监控 |

完整交互式文档：启动后访问 http://127.0.0.1:8000/docs

## 兼容性

- 标注文件格式与桌面版**逐字段一致**（含矩形四点、rotation direction、cuboid3d 元数据），双向可互开
- 训练历史与桌面版共享同一存储目录
- 桌面版代码零改动，Web 版全部位于 `web/` 目录
