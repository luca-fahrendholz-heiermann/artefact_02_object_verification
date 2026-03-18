import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import open3d as o3d
import pandas as pd

MAX_VIEWER_POINTS = 12000


def _load_df(prediction_file: Path) -> pd.DataFrame:
    if prediction_file.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(prediction_file)
    return pd.read_json(prediction_file)


def _safe_points(path_str: str) -> np.ndarray:
    if not path_str or not Path(path_str).exists():
        return np.zeros((0, 3), dtype=np.float32)
    pcd = o3d.io.read_point_cloud(path_str)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.shape[0] > MAX_VIEWER_POINTS:
        idx = np.random.choice(pts.shape[0], size=MAX_VIEWER_POINTS, replace=False)
        pts = pts[idx]
    return pts


def _threejs_html(point_sets: List[np.ndarray], colors: List[str], title: str, height: int = 280) -> str:
    arrays = []
    for pts, color in zip(point_sets, colors):
        if pts.size == 0:
            continue
        arrays.append({"pts": pts.tolist(), "color": color})

    payload = json.dumps(arrays)
    dom_id = f"v_{abs(hash(title + str(len(payload)))) % (10**8)}"
    return f"""
    <div style='font-family:Arial; font-size:13px; margin-bottom:6px;'><b>{title}</b></div>
    <div id='{dom_id}' style='width:100%; height:{height}px; border:1px solid #ddd;'></div>
    <script src='https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js'></script>
    <script src='https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js'></script>
    <script>
      (function() {{
        const root = document.getElementById('{dom_id}');
        root.innerHTML = '';
        const data = {payload};
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf7f7f7);
        const camera = new THREE.PerspectiveCamera(65, root.clientWidth/root.clientHeight, 0.01, 2000);
        const renderer = new THREE.WebGLRenderer({{antialias:true}});
        renderer.setSize(root.clientWidth, root.clientHeight);
        root.appendChild(renderer.domElement);
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        let center = new THREE.Vector3(0,0,0);
        let pointsCount = 0;
        data.forEach(cloud => {{
          const arr = new Float32Array(cloud.pts.flat());
          const geo = new THREE.BufferGeometry();
          geo.setAttribute('position', new THREE.BufferAttribute(arr, 3));
          const mat = new THREE.PointsMaterial({{size:0.03, color:new THREE.Color(cloud.color)}});
          const ptsObj = new THREE.Points(geo, mat);
          scene.add(ptsObj);
          const box = new THREE.Box3().setFromBufferAttribute(geo.getAttribute('position'));
          center.add(box.getCenter(new THREE.Vector3()));
          pointsCount += 1;
        }});
        center = pointsCount > 0 ? center.divideScalar(pointsCount) : center;
        camera.position.set(center.x + 2, center.y + 2, center.z + 2);
        controls.target.copy(center);
        controls.update();
        scene.add(new THREE.AxesHelper(0.5));
        const light = new THREE.HemisphereLight(0xffffff, 0x444444, 1.2);
        scene.add(light);
        const animate = () => {{ requestAnimationFrame(animate); controls.update(); renderer.render(scene,camera); }};
        animate();
      }})();
    </script>
    """


def _build_viewers(row: pd.Series) -> Tuple[str, str, str, str, str, str, str]:
    source = _safe_points(str(row.get("source_file", "")))
    target = _safe_points(str(row.get("reference_file", "")))
    extracted = _safe_points(str(row.get("extracted_file", "")))

    viewer1 = _threejs_html([target], ["#222222"], "1) Referenz (Target oder As-Planned Part)")
    viewer2 = _threejs_html([source], ["#8b8b8b"], "2) As-Built Scan")
    viewer3 = _threejs_html([source, target], ["#8b8b8b", "#00ff00"], "3) Scan + As-Planned Overlay")
    viewer4 = _threejs_html([target], ["#4444ff"], "4) As-Planned Part ID")
    viewer5 = _threejs_html([target, extracted], ["#4444ff", "#00cc00"], "5) As-Built Part vs As-Planned Part")
    viewer6 = _threejs_html([extracted], ["#00cc00"], "6) As-Built Part (extracted)")

    meta = (
        f"**Projekt:** {row.get('project')}  \\n"
        f"**Scope:** {row.get('comparison_scope')}  \\n"
        f"**Source:** {row.get('source_file')}  \\n"
        f"**Prediction:** class={row.get('predicted_class')} | p1={row.get('probability_class_1', 0):.6f} | "
        f"threshold={row.get('threshold_used', 0):.2f}"
    )
    return viewer1, viewer2, viewer3, viewer4, viewer5, viewer6, meta


def launch(prediction_file: Path, server_port: int = 7860) -> None:
    df = _load_df(prediction_file)
    if "ground_truth" not in df.columns:
        df["ground_truth"] = np.nan

    projects = sorted(df["project"].dropna().unique().tolist())

    def scans_for_project(project: str):
        rows = df[df["project"] == project]
        scans = sorted(rows["source_file"].dropna().unique().tolist())
        return gr.Dropdown(choices=scans, value=scans[0] if scans else None)

    def rows_for_project_scan(project: str, scan: str):
        rows = df[(df["project"] == project) & (df["source_file"] == scan)]
        idxs = rows.index.tolist()
        labels = [f"idx={idx} | {rows.loc[idx, 'comparison_scope']} | {Path(str(rows.loc[idx, 'reference_file'])).name}" for idx in idxs]
        default = idxs[0] if idxs else None
        return gr.Dropdown(choices=[(labels[i], idxs[i]) for i in range(len(idxs))], value=default)

    def show_row(idx: int):
        if idx is None:
            empty = "<div>Keine Daten</div>"
            return empty, empty, empty, empty, empty, empty, "Keine Zeile gewählt", None
        row = df.loc[idx]
        v1, v2, v3, v4, v5, v6, meta = _build_viewers(row)
        gt = None if pd.isna(row.get("ground_truth")) else int(row.get("ground_truth"))
        return v1, v2, v3, v4, v5, v6, meta, gt

    def save_gt(idx: int, gt: int):
        if idx is None or gt is None:
            return "Bitte Zeile und Ground Truth wählen."
        df.loc[idx, "ground_truth"] = int(gt)
        if prediction_file.suffix.lower() in {".xlsx", ".xls"}:
            df.to_excel(prediction_file, index=False)
        else:
            df.to_json(prediction_file, orient="records", indent=2, force_ascii=False)
        return f"Gespeichert: idx={idx}, ground_truth={gt}"

    with gr.Blocks(title="Validation Labeling Interface") as app:
        gr.Markdown("## Object Verification Labeling Interface")
        with gr.Row():
            dd_project = gr.Dropdown(choices=projects, value=projects[0] if projects else None, label="Projekt")
            dd_scan = gr.Dropdown(label="Source Scan")
            dd_row = gr.Dropdown(label="Vergleichszeile")

        with gr.Row():
            v1 = gr.HTML()
        with gr.Row():
            v2 = gr.HTML()
            v3 = gr.HTML()
        with gr.Row():
            v4 = gr.HTML()
            v5 = gr.HTML()
            v6 = gr.HTML()

        meta = gr.Markdown()
        gt = gr.Radio(choices=[0, 1], label="Ground Truth (0/1)")
        btn = gr.Button("Ground Truth speichern")
        status = gr.Textbox(label="Status")

        dd_project.change(scans_for_project, inputs=[dd_project], outputs=[dd_scan])
        dd_scan.change(rows_for_project_scan, inputs=[dd_project, dd_scan], outputs=[dd_row])
        dd_row.change(show_row, inputs=[dd_row], outputs=[v1, v2, v3, v4, v5, v6, meta, gt])
        btn.click(save_gt, inputs=[dd_row, gt], outputs=[status])

        app.load(scans_for_project, inputs=[dd_project], outputs=[dd_scan]).then(
            rows_for_project_scan, inputs=[dd_project, dd_scan], outputs=[dd_row]
        ).then(show_row, inputs=[dd_row], outputs=[v1, v2, v3, v4, v5, v6, meta, gt])

    app.launch(server_name="0.0.0.0", server_port=server_port)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch gradio interface for ground-truth labeling.")
    parser.add_argument("--prediction-file", required=True)
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    launch(Path(args.prediction_file), server_port=args.port)
