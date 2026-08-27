import axios from "axios";
import type {
  GetLabelsResponse,
  LabelFileData,
  OpenDirResponse,
  Shape,
} from "../types";

const api = axios.create({ baseURL: "/api" });

// ---- access token + remote server (remote mode) ------------------------------

const TOKEN_KEY = "xaw_token";
const SERVER_KEY = "xaw_server";

export function getToken(): string {
  return sessionStorage.getItem(TOKEN_KEY) ?? "";
}

export function setToken(token: string): void {
  sessionStorage.setItem(TOKEN_KEY, token);
}

/** Remote backend base (e.g. http://gpu-server:8000). "" = same origin. */
export function getServerUrl(): string {
  return localStorage.getItem(SERVER_KEY) ?? "";
}

export function setServerUrl(url: string): void {
  const u = url.trim().replace(/\/+$/, "");
  if (u) localStorage.setItem(SERVER_KEY, u);
  else localStorage.removeItem(SERVER_KEY);
}

api.interceptors.request.use((cfg) => {
  cfg.baseURL = `${getServerUrl()}/api`;
  const t = getToken();
  if (t) cfg.headers.Authorization = `Bearer ${t}`;
  return cfg;
});

api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err?.response?.status === 401) {
      window.dispatchEvent(new Event("xaw:unauthorized"));
    }
    return Promise.reject(err);
  }
);

/** Resolve a server-relative API path for non-axios consumers (<img>, downloads). */
function withToken(url: string): string {
  const full = `${getServerUrl()}${url}`;
  const t = getToken();
  if (!t) return full;
  return full + (full.includes("?") ? "&" : "?") + `token=${encodeURIComponent(t)}`;
}

export async function openDir(path: string): Promise<OpenDirResponse> {
  const r = await api.post<OpenDirResponse>("/dir/open", { path });
  return r.data;
}

export async function getDirImages(): Promise<OpenDirResponse> {
  const r = await api.get<OpenDirResponse>("/dir/images");
  return r.data;
}

export async function getLabels(image: string): Promise<GetLabelsResponse> {
  const r = await api.get<GetLabelsResponse>("/labels", {
    params: { image },
  });
  return r.data;
}

export interface SaveLabelsPayload {
  image: string;
  shapes: Shape[];
  flags: Record<string, unknown>;
  other_data: Record<string, unknown>;
  image_height: number;
  image_width: number;
}

export async function putLabels(payload: SaveLabelsPayload): Promise<void> {
  await api.put("/labels", payload);
}

export async function deleteLabels(image: string): Promise<void> {
  await api.delete("/labels", { params: { image } });
}

export function imageUrl(filename: string): string {
  return withToken(`/api/image?path=${encodeURIComponent(filename)}`);
}

export function thumbUrl(filename: string): string {
  return withToken(`/api/image/thumb?path=${encodeURIComponent(filename)}`);
}

// ---- filesystem browsing ---------------------------------------------------

export interface FsDirEntry {
  name: string;
  path: string;
  has_images: boolean;
}

export interface FsListResponse {
  path: string;
  parent: string | null;
  dirs: FsDirEntry[];
  files?: { name: string; path: string }[];
  has_images: boolean;
  roots?: string[];
}

export async function fsList(path?: string, withFiles?: string): Promise<FsListResponse> {
  const r = await api.get("/fs/list", {
    params: { path: path ?? "", with_files: withFiles ?? "" },
  });
  return r.data;
}

export async function health(): Promise<{ status: string }> {
  const r = await api.get("/health");
  return r.data;
}

// ---- auto labeling ---------------------------------------------------------

export interface ModelInfo {
  name: string;
  display_name: string;
  type: string;
  provider?: string;
  config_file: string;
  is_custom_model: boolean;
}

export interface LoadedModelInfo {
  name: string;
  display_name: string;
  type: string;
  config_file: string;
  supports_marks?: boolean;
  output_modes?: string[];
  default_output_mode?: string;
  widgets?: string[];
}

export interface ModelStatus {
  loading: boolean;
  error: string | null;
  progress: { downloaded: number; total: number } | null;
  message: string;
  loaded: LoadedModelInfo | null;
}

export async function getModels(): Promise<{
  models: ModelInfo[];
  loaded: LoadedModelInfo | null;
}> {
  const r = await api.get("/models");
  return r.data;
}

export async function loadModel(configFile: string): Promise<void> {
  await api.post("/models/load", { config_file: configFile });
}

export async function unloadModel(): Promise<void> {
  await api.post("/models/unload");
}

export async function setOutputMode(mode: string): Promise<void> {
  await api.post("/models/output_mode", { mode });
}

export async function getModelStatus(): Promise<ModelStatus> {
  const r = await api.get("/models/status");
  return r.data;
}

export interface RegisterLocalModelResult {
  registered: boolean;
  config_file: string;
  display_name: string;
  model_type: string;
}

export async function registerLocalModel(payload: {
  template_config_file: string;
  local_path: string;
  display_name?: string;
}): Promise<RegisterLocalModelResult> {
  const r = await api.post("/models/register-local", payload);
  return r.data;
}

export async function deleteCustomModel(
  configFile: string,
  deleteFile = false
): Promise<{ deleted: boolean; deleted_file: boolean }> {
  const r = await api.delete("/models/custom", {
    params: { config_file: configFile, delete_file: deleteFile },
  });
  return r.data;
}

export interface LocalModelFileInfo {
  config_file: string;
  name: string;
  display_name: string;
  type: string;
  is_custom_model: boolean;
  downloaded: boolean;
  path: string | null;
  size_bytes: number;
}

export interface LocalModelFilesReport {
  root: string;
  items: LocalModelFileInfo[];
  total_bytes: number;
}

export async function getLocalModelFiles(): Promise<LocalModelFilesReport> {
  const r = await api.get("/models/local-files");
  return r.data;
}

export async function deleteModelCache(
  configFile: string
): Promise<{ deleted: boolean; freed_bytes: number }> {
  const r = await api.delete("/models/cache", { params: { config_file: configFile } });
  return r.data;
}

export interface ScannedModelFile {
  path: string;
  name: string;
  size_bytes: number;
  modified_at: string;
}

export async function scanModelDir(path: string): Promise<{ files: ScannedModelFile[] }> {
  const r = await api.get("/models/scan-dir", { params: { path } });
  return r.data;
}

export interface PredictResult {
  shapes: Shape[];
  replace: boolean;
  description: string;
}

export async function predict(
  image: string,
  textPrompt?: string,
  conf?: number,
  iou?: number
): Promise<PredictResult> {
  const r = await api.post("/predict", {
    image,
    text_prompt: textPrompt || null,
    conf: conf ?? null,
    iou: iou ?? null,
  });
  return r.data;
}

export interface SamMark {
  type: "point" | "rectangle";
  data: number[];
  label?: number;
}

export async function predictSam(
  image: string,
  marks: SamMark[]
): Promise<PredictResult> {
  const r = await api.post("/predict/sam", { image, marks });
  return r.data;
}

export async function predictBatch(
  images: string[],
  preserveExisting: boolean,
  conf?: number,
  iou?: number,
  textPrompt?: string
): Promise<void> {
  await api.post("/predict/batch", {
    images,
    preserve_existing: preserveExisting,
    conf: conf ?? null,
    iou: iou ?? null,
    text_prompt: textPrompt || null,
  });
}

export interface BatchStatus {
  running: boolean;
  current?: number;
  total?: number;
  current_image?: string | null;
  errors?: { image: string; error: string }[];
  undo_available?: boolean;
  backup_count?: number;
  batch_id?: string | null;
}

export async function getBatchStatus(): Promise<BatchStatus> {
  const r = await api.get("/predict/batch/status");
  return r.data;
}

export interface UndoBatchResult {
  restored: number;
  deleted: number;
  already_missing: number;
  skipped_modified: string[];
}

export async function undoBatch(): Promise<UndoBatchResult> {
  const r = await api.post("/predict/batch/undo");
  return r.data;
}

// ---- playground (one-off inference on an uploaded image) ----------------------

export interface PlaygroundPredictResult {
  shapes: Shape[];
  model: { display_name: string; type: string };
}

export async function playgroundPredict(
  file: File,
  conf?: number,
  iou?: number
): Promise<PlaygroundPredictResult> {
  const form = new FormData();
  form.append("file", file);
  if (conf != null) form.append("conf", String(conf));
  if (iou != null) form.append("iou", String(iou));
  const r = await api.post("/playground/predict", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return r.data;
}

// ---- active learning (hard-example scanning) ------------------------------------

export interface ALScanStartResult {
  started: boolean;
  total: number;
}

export interface ALScanStatus {
  running: boolean;
  processed: number;
  total: number;
  done: boolean;
  error: string | null;
  error_count: number;
}

export interface ALScoreEntry {
  score: number | null;
  count: number;
}

export async function startALScan(
  conf?: number,
  scope: "unlabeled" | "all" = "unlabeled"
): Promise<ALScanStartResult> {
  const r = await api.post("/active_learning/scan", { conf: conf ?? null, scope });
  return r.data;
}

export async function getALScanStatus(): Promise<ALScanStatus> {
  const r = await api.get("/active_learning/scan/status");
  return r.data;
}

export async function stopALScan(): Promise<{ stopping: boolean }> {
  const r = await api.post("/active_learning/scan/stop");
  return r.data;
}

export async function getALScores(): Promise<{
  scores: Record<string, ALScoreEntry>;
  updated_at: string | null;
}> {
  const r = await api.get("/active_learning/scores");
  return r.data;
}

export async function clearALScores(): Promise<{ cleared: boolean }> {
  const r = await api.post("/active_learning/clear");
  return r.data;
}

// ---- export -----------------------------------------------------------------

export interface ExportFormatInfo {
  modes: string[];
  default_mode: string | null;
}

export interface ExportRequestPayload {
  format: string;
  mode?: string;
  output_dir: string;
  save_images?: boolean;
  skip_empty_files?: boolean;
}

export interface ExportResult {
  output_dir: string;
  files_written: number;
  format: string;
  mode: string | null;
  classes: string[];
}

export interface ExportStatus {
  running: boolean;
  current: number;
  total: number;
  message: string;
  result: ExportResult | null;
  error: string | null;
}

export async function getExportFormats(): Promise<{
  formats: Record<string, ExportFormatInfo>;
}> {
  const r = await api.get("/export/formats");
  return r.data;
}

export async function startExport(payload: ExportRequestPayload): Promise<void> {
  await api.post("/export", payload);
}

export async function getExportStatus(): Promise<ExportStatus> {
  const r = await api.get("/export/status");
  return r.data;
}

export function exportDownloadUrl(outputDir: string): string {
  return withToken(`/api/export/download?path=${encodeURIComponent(outputDir)}`);
}

// ---- upload -----------------------------------------------------------------

export async function uploadFiles(files: File[]): Promise<{
  saved: number;
  skipped: string[];
  total_images: number;
}> {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  const r = await api.post("/upload/files", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return r.data;
}

// ---- video ------------------------------------------------------------------

export interface VideoInfo {
  video: string;
  frame_count: number;
  fps: number;
  width: number;
  height: number;
  labeled_frames: number[];
}

export async function openVideo(path: string): Promise<VideoInfo> {
  const r = await api.post("/video/open", { path });
  return r.data;
}

export async function closeVideo(): Promise<void> {
  await api.post("/video/close");
}

export async function getVideoInfo(): Promise<VideoInfo> {
  const r = await api.get("/video/info");
  return r.data;
}

export function videoFrameUrl(index: number): string {
  return withToken(`/api/video/frame?index=${index}&t=${Date.now()}`);
}

export async function getVideoLabels(index: number): Promise<GetLabelsResponse> {
  const r = await api.get("/video/labels", { params: { index } });
  return r.data;
}

export async function putVideoLabels(
  index: number,
  payload: Omit<SaveLabelsPayload, "image">
): Promise<void> {
  await api.put("/video/labels", { ...payload, image: String(index) });
}

export interface TrackRequestPayload {
  start_frame?: number;
  end_frame?: number;
  conf?: number;
  iou?: number;
  preserve_existing?: boolean;
}

export async function startTrack(payload: TrackRequestPayload): Promise<void> {
  await api.post("/video/track", payload);
}

export interface TrackStatus {
  running: boolean;
  current: number;
  total: number;
  current_frame: number | null;
  errors: { frame: number; error: string }[];
  result: { frames: number; errors: number } | null;
  undo_available?: boolean;
}

export async function getTrackStatus(): Promise<TrackStatus> {
  const r = await api.get("/video/track/status");
  return r.data;
}

// ---- training center ----------------------------------------------------------

export interface GuidedStartPayload {
  task: string;
  model: string;
  data: string;
  project: string;
  name: string;
  device: string;
  epochs?: number;
  batch?: number;
  imgsz?: number;
  patience?: number;
  lr0?: number;
  execution_mode?: string;
  remote_profile_id?: string | null;
  remote_password?: string | null;
  [key: string]: unknown;
}

export interface TrainingJobInfo {
  job_id: string;
  mode: string;
  status: string;
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
  display_name: string;
  error_message: string | null;
  [key: string]: unknown;
}

export interface TrainingStatusResponse {
  job: TrainingJobInfo | null;
  running: boolean;
  output_dir: string | null;
  eta_seconds?: number | null;
}

export interface TrainingEventItem {
  seq: number;
  event_type: string;
  job_id?: string;
  payload?: Record<string, unknown>;
}

export interface MetricSeries {
  name: string;
  group: string;
  points: [number, number][];
}

export interface PreflightIssue {
  severity: string;
  title: string;
  message: string;
  suggestion: string | null;
}

export async function guidedStart(payload: GuidedStartPayload): Promise<TrainingStatusResponse> {
  const r = await api.post("/training/guided/start", payload);
  return r.data;
}

export async function trainingStop(): Promise<void> {
  await api.post("/training/stop");
}

export async function getTrainingStatus(): Promise<TrainingStatusResponse> {
  const r = await api.get("/training/status");
  return r.data;
}

export async function getTrainingEvents(
  since: number
): Promise<{ latest: number; events: TrainingEventItem[] }> {
  const r = await api.get("/training/events", { params: { since } });
  return r.data;
}

export async function getTrainingMetrics(): Promise<{ series: MetricSeries[] }> {
  const r = await api.get("/training/metrics");
  return r.data;
}

export async function getTrainingHistory(limit = 50): Promise<{ jobs: Record<string, unknown>[] }> {
  const r = await api.get("/training/history", { params: { limit } });
  return r.data;
}

export async function trainingPreflight(payload: Record<string, unknown>): Promise<{
  can_start: boolean;
  issues: PreflightIssue[];
}> {
  const r = await api.post("/training/preflight", payload);
  return r.data;
}

// ---- remote SSH training servers ---------------------------------------------

export interface RemoteProfile {
  profile_id: string;
  name: string;
  host: string;
  port: number;
  username: string;
  auth_method: "ssh_key" | "password";
  private_key_path: string;
  remote_workspace: string;
  remote_python: string;
  proxy?: string;
  known_host_fingerprint: string;
  created_at?: string;
  updated_at?: string;
}

export interface RemoteProfilePayload {
  name: string;
  host: string;
  port: number;
  username: string;
  auth_method: "ssh_key" | "password";
  private_key_path?: string;
  remote_workspace: string;
  remote_python: string;
  proxy?: string;
}

export interface RemoteTestResult {
  ok: boolean;
  need_host_key_confirm?: boolean;
  fingerprint?: string;
  error?: string;
}

export interface RemoteDiagnosticItem {
  stage: string;
  label: string;
  status: string; // PASS / WARNING / ERROR
  message: string;
  details?: Record<string, unknown>;
}

export interface RemoteGpuInfo {
  index: number;
  name: string;
  total_memory_gb: number;
  compute_capability?: string;
}

export interface RemoteDiagnosticsResult {
  items: RemoteDiagnosticItem[];
  recommended_device: string; // "0" | "cpu"
  gpus: RemoteGpuInfo[];
}

export async function listRemoteProfiles(): Promise<{ profiles: RemoteProfile[] }> {
  const r = await api.get("/remote/profiles");
  return r.data;
}

export async function createRemoteProfile(
  payload: RemoteProfilePayload
): Promise<{ profile: RemoteProfile }> {
  const r = await api.post("/remote/profiles", payload);
  return r.data;
}

export async function updateRemoteProfile(
  profileId: string,
  payload: RemoteProfilePayload
): Promise<{ profile: RemoteProfile }> {
  const r = await api.put(`/remote/profiles/${encodeURIComponent(profileId)}`, payload);
  return r.data;
}

export async function deleteRemoteProfile(profileId: string): Promise<void> {
  await api.delete(`/remote/profiles/${encodeURIComponent(profileId)}`);
}

export async function testRemoteProfile(
  profileId: string,
  password?: string
): Promise<RemoteTestResult> {
  const r = await api.post(`/remote/profiles/${encodeURIComponent(profileId)}/test`, {
    password: password ?? null,
  });
  return r.data;
}

export async function confirmRemoteHostKey(
  profileId: string,
  fingerprint: string
): Promise<{ saved: boolean }> {
  const r = await api.post(
    `/remote/profiles/${encodeURIComponent(profileId)}/confirm-host-key`,
    { fingerprint }
  );
  return r.data;
}

export async function remoteDiagnostics(
  profileId: string,
  password?: string
): Promise<RemoteDiagnosticsResult> {
  const r = await api.post(
    `/remote/profiles/${encodeURIComponent(profileId)}/diagnostics`,
    { password: password ?? null }
  );
  return r.data;
}

// ---- run monitor --------------------------------------------------------------

export interface DetectedScriptInfo {
  path: string;
  framework: string | null;
  confidence: number;
  reasons: string[];
}

export interface PythonEnvInfo {
  python_path: string;
  version: string;
  env_type: string;
  is_valid: boolean;
}

export interface WorkspaceInfo {
  path: string;
  detected_scripts: DetectedScriptInfo[];
  detected_environments: PythonEnvInfo[];
}

export interface RunInfo {
  run_id: string;
  script_path: string;
  python_path: string;
  arguments: string;
  pid: number;
  status: string;
  started_at: number;
  ended_at: number | null;
  exit_code: number | null;
}

export interface ResourceSample {
  ts: number;
  system_cpu: number;
  system_mem_percent: number;
  proc_cpu?: number;
  proc_rss_mb?: number;
  gpu_util?: number;
  gpu_mem_used_mb?: number;
}

export async function monitorScan(workspace: string): Promise<WorkspaceInfo> {
  const r = await api.post("/monitor/scan", { workspace });
  return r.data;
}

export async function monitorStart(payload: {
  workspace: string;
  script_path: string;
  python_path: string;
  arguments: string;
}): Promise<RunInfo> {
  const r = await api.post("/monitor/start", payload);
  return r.data;
}

export async function monitorStop(): Promise<{ stopped: boolean; reason?: string }> {
  const r = await api.post("/monitor/stop");
  return r.data;
}

export async function monitorStatus(): Promise<{ running: boolean; run: RunInfo | null }> {
  const r = await api.get("/monitor/status");
  return r.data;
}

export async function monitorLogs(
  since: number
): Promise<{ latest: number; lines: { seq: number; stream: string; line: string }[] }> {
  const r = await api.get("/monitor/logs", { params: { since } });
  return r.data;
}

export async function monitorResources(limit = 300): Promise<{ samples: ResourceSample[] }> {
  const r = await api.get("/monitor/resources", { params: { limit } });
  return r.data;
}

// ---- export trained model -----------------------------------------------------

export interface ExportModelResult {
  exported: boolean;
  relative_path: string;
  format: string;
}

export async function exportModelArtifact(
  jobId: string,
  path: string,
  format: string
): Promise<ExportModelResult> {
  const r = await api.post(
    `/training/history/${encodeURIComponent(jobId)}/artifacts/export`,
    { path, format }
  );
  return r.data;
}

export interface RegisterModelResult {
  registered: boolean;
  config_file: string;
  display_name: string;
  model_type: string;
}

export async function registerModelArtifact(
  jobId: string,
  path: string,
  displayName?: string
): Promise<RegisterModelResult> {
  const r = await api.post(
    `/training/history/${encodeURIComponent(jobId)}/artifacts/register-model`,
    { path, display_name: displayName ?? null }
  );
  return r.data;
}

export interface TrainingExportFormat {
  id: string;
  name: string;
  available: boolean;
  reason: string | null;
}

export async function getTrainingExportFormats(): Promise<{
  formats: TrainingExportFormat[];
}> {
  const r = await api.get("/training/export-formats");
  return r.data;
}

export interface PrepareDatasetPayload {
  task_type: string;
  dataset_ratio: number;
  skip_empty_files?: boolean;
  only_checked_files?: boolean;
}

export interface PrepareDatasetResult {
  dataset_dir: string;
  data_yaml: string;
  info: string;
}

export interface DatasetStatsWarning {
  code: string;
  message: string;
}

export interface DatasetStats {
  total_images: number;
  per_task_valid: Record<string, number>;
  label_infos: Record<string, Record<string, number>>;
  class_counts: Record<string, number>;
  warnings: DatasetStatsWarning[];
  recommended_task: string;
}

export async function getDatasetStats(): Promise<DatasetStats> {
  const r = await api.get("/dataset/stats");
  return r.data;
}

export async function prepareDataset(
  payload: PrepareDatasetPayload
): Promise<PrepareDatasetResult> {
  const r = await api.post("/dataset/prepare", payload);
  return r.data;
}

// ---- dataset health check -------------------------------------------------------

export interface HealthScanStartResult {
  started: boolean;
  total: number;
}

export interface HealthScanStatus {
  running: boolean;
  processed: number;
  total: number;
  done: boolean;
  error: string | null;
  error_count: number;
}

export interface HealthShapeIssue {
  label: string;
  issue: string; // out_of_bounds / oversized / tiny / degenerate
  detail: string;
}

export interface DatasetHealthSummary {
  total_images: number;
  duplicate_groups: number;
  duplicate_images: number;
  blurry: number;
  dark: number;
  bright: number;
  shape_issue_images: number;
}

// 报告不存在时后端只返回 { updated_at: null }，其余字段全部缺省
export interface DatasetHealthReport {
  summary?: DatasetHealthSummary;
  duplicate_groups?: string[][];
  blurry?: { name: string; score: number }[];
  dark?: { name: string; brightness: number }[];
  bright?: { name: string; brightness: number }[];
  shape_issues?: { name: string; issues: HealthShapeIssue[] }[];
  updated_at: string | null;
}

export async function startHealthScan(): Promise<HealthScanStartResult> {
  const r = await api.post("/dataset/health/scan", {});
  return r.data;
}

export async function getHealthScanStatus(): Promise<HealthScanStatus> {
  const r = await api.get("/dataset/health/scan/status");
  return r.data;
}

export async function stopHealthScan(): Promise<{ stopping: boolean }> {
  const r = await api.post("/dataset/health/scan/stop");
  return r.data;
}

export async function getDatasetHealthReport(): Promise<DatasetHealthReport> {
  const r = await api.get("/dataset/health/report");
  return r.data;
}

// ---- device + quickstart --------------------------------------------------------

export interface DeviceInfo {
  cuda_available: boolean;
  gpus: { index: number; name: string; memory_mb: number }[];
  recommended: string;
}

export async function getDevice(): Promise<DeviceInfo> {
  const r = await api.get("/system/device");
  return r.data;
}

export async function getDemoDir(): Promise<{ exists: boolean; path: string | null }> {
  const r = await api.get("/system/demo-dir");
  return r.data;
}

export interface InferenceDeviceInfo {
  current: string;  // "CPU" | "GPU" | "auto"
  available: string[];  // ["CPU"] or ["CPU", "GPU"]
}

export async function getInferenceDevice(): Promise<InferenceDeviceInfo> {
  const r = await api.get("/system/device/inference");
  return r.data;
}

export async function setInferenceDevice(device: string): Promise<{ ok: boolean; current: string }> {
  const r = await api.post("/system/device/inference", { device });
  return r.data;
}

export interface ArtifactInfo {
  name: string;
  relative_path: string;
  size: number;
  modified_at: string;
  is_downloadable: boolean;
  file_type: string;
}

export interface ArtifactListResponse {
  job_id: string;
  output_dir: string;
  artifacts: ArtifactInfo[];
}

export async function getTrainingArtifacts(jobId: string): Promise<ArtifactListResponse> {
  const r = await api.get(`/training/history/${encodeURIComponent(jobId)}/artifacts`);
  return r.data;
}

export function trainingArtifactDownloadUrl(jobId: string, path: string): string {
  return withToken(
    `/api/training/history/${encodeURIComponent(jobId)}/artifacts/download?path=${encodeURIComponent(path)}`
  );
}

export function trainingArtifactDownloadAllUrl(jobId: string): string {
  return withToken(`/api/training/history/${encodeURIComponent(jobId)}/artifacts/download-all`);
}

export interface QuickstartResult {
  task_type: string;
  device: string;
  model: string;
  dataset_dir: string;
  dataset_info: string;
  job: TrainingJobInfo;
}

export interface QuickstartPayload {
  epochs?: number;
}

export async function quickstart(payload: QuickstartPayload): Promise<QuickstartResult> {
  const r = await api.post("/training/quickstart", payload);
  return r.data;
}

export type { LabelFileData };
