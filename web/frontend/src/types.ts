/** Types mirroring the desktop labelme-style annotation JSON. */

export type ShapeType =
  | "rectangle"
  | "polygon"
  | "rotation"
  | "circle"
  | "line"
  | "point"
  | "linestrip"
  | "cuboid";

export type Point = [number, number];

export interface Shape {
  label: string;
  points: Point[];
  shape_type: ShapeType;
  group_id?: number | null;
  description?: string;
  difficult?: boolean;
  score?: number | null;
  flags?: Record<string, unknown>;
  attributes?: Record<string, unknown>;
  kie_linking?: unknown[];
  direction?: number;
  locked?: boolean;
  cuboid3d?: {
    depth_vector: [number, number];
    mode: string;
    source: string;
  };
}

export interface LabelFileData {
  version?: string;
  flags: Record<string, unknown>;
  checked: boolean;
  shapes: Shape[];
  imagePath: string;
  imageData?: string | null;
  imageHeight: number;
  imageWidth: number;
  [key: string]: unknown;
}

export interface ImageInfo {
  filename: string;
  has_label: boolean;
  shape_count: number | null; // null = no label file
  labels?: string[];
  label_counts?: Record<string, number>;
}

export interface OpenDirResponse {
  dir: string;
  images: ImageInfo[];
}

export interface GetLabelsResponse {
  exists: boolean;
  data: LabelFileData | null;
}
