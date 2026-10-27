#!/usr/bin/env python3
"""Film de la simulation : séparation vérité terrain vs détection tracker."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar_tracker.config import load_config
from radar_tracker.coordinates import spherical_to_cartesian
from radar_tracker.csv_reader import DetectionRecord, ScanBatch
from radar_tracker.simulator.sensor import simulate_scan
from radar_tracker.simulator.truth import generate_scenario_truth
from radar_tracker.tracker import RadarSeparationTracker


@dataclass
class FrameState:
    scan_index: int
    timestamp_str: str
    truth_xy: dict[str, tuple[float, float]]
    det_xy: list[tuple[float, float]]
    track_xy: dict[str, tuple[float, float]]
    n_dets: int


def build_frames(cfg, scenario: str, seed: int) -> tuple[list[FrameState], dict, list]:
    duration = float(cfg.simulation.get("duration_s", 120))
    interval = float(cfg.simulation.get("scan_interval_s", 1.0))
    truth = generate_scenario_truth(scenario, duration, interval, seed=seed)
    rng = np.random.default_rng(seed)
    n_scans = int(duration / interval) + 1
    radar_pos = np.array(cfg.radar.position_m, dtype=float)

    gt_sep_scan = None
    if truth.events:
        t_sep = truth.events[0].time
        gt_sep_scan = int(round((t_sep - truth.start_time).total_seconds() / interval))

    tracker = RadarSeparationTracker(cfg)
    frames: list[FrameState] = []

    for si in range(n_scans):
        sim_dets = simulate_scan(truth, si, cfg, rng)
        batch = ScanBatch(
            timestamp=sim_dets[0].timestamp if sim_dets else truth.start_time + timedelta(seconds=si),
            radar_id=cfg.radar.id,
            scan_id=f"S{si:05d}",
            detections=[
                DetectionRecord(
                    timestamp=d.timestamp,
                    radar_id=d.radar_id,
                    scan_id=d.scan_id,
                    detection_id=d.detection_id,
                    range_m=d.range_m,
                    azimuth_rad=d.azimuth_rad,
                    elevation_rad=d.elevation_rad,
                )
                for d in sim_dets
            ],
        )
        tracker.process_scan(batch)

        truth_xy = {}
        ts = batch.timestamp
        for tgt in truth.targets:
            if ts < tgt.birth_time:
                continue
            for i, tts in enumerate(tgt.timestamps):
                if tts == ts:
                    s = tgt.states[i]
                    truth_xy[tgt.target_id] = (float(s[0]), float(s[2]))
                    break

        det_xy = [
            tuple(spherical_to_cartesian(d.range_m, d.azimuth_rad, d.elevation_rad, radar_pos)[:2])
            for d in batch.detections
        ]
        track_xy = {}
        for tid, tr in tracker.tracks.items():
            if tr.state is not None:
                sv = tr.state.state_vector.ravel()
                track_xy[tid] = (float(sv[0]), float(sv[2]))

        frames.append(
            FrameState(
                scan_index=si,
                timestamp_str=batch.timestamp.strftime("%H:%M:%S"),
                truth_xy=truth_xy,
                det_xy=det_xy,
                track_xy=track_xy,
                n_dets=len(batch.detections),
            )
        )

    det_sep_scan = None
    if tracker.events and gt_sep_scan is not None:
        from datetime import datetime

        conf = tracker.events[0].confirmation_time
        if conf:
            tc = datetime.fromisoformat(conf)
            det_sep_scan = int(round((tc - truth.start_time).total_seconds() / interval))

    meta = {
        "gt_separation_scan": gt_sep_scan,
        "detection_scan": det_sep_scan,
        "delay_s": (det_sep_scan - gt_sep_scan) if det_sep_scan is not None and gt_sep_scan is not None else None,
        "gt_separation_time": truth.events[0].time.isoformat() if truth.events else None,
        "events": [e.to_dict() for e in tracker.events],
    }
    return frames, meta, truth


def render_mp4(frames: list[FrameState], meta: dict, output: Path, fps: float = 2.0) -> None:
    gt_scan = meta.get("gt_separation_scan")
    det_scan = meta.get("detection_scan")

    # accumulate truth trails
    trails: dict[str, list[tuple[float, float]]] = {"AC": [], "MS": []}
    parent_trail: list[tuple[float, float]] = []
    child_trail: list[tuple[float, float]] = []

    all_x = []
    all_y = []
    for fr in frames:
        for xy in fr.det_xy + list(fr.truth_xy.values()) + list(fr.track_xy.values()):
            all_x.append(xy[0])
            all_y.append(xy[1])
    pad = 2000
    xmin, xmax = min(all_x) - pad, max(all_x) + pad
    ymin, ymax = min(all_y) - pad, max(all_y) + pad

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#0f1419")

    (line_t1,) = ax.plot([], [], color="#4fc3f7", lw=2, label="Avion porteur (AC)")
    (line_t2,) = ax.plot([], [], color="#ff8a65", lw=2, label="Missile air-mer (MS)")
    (line_p,) = ax.plot([], [], color="#aed581", lw=2, ls="--", label="Piste estimée parent")
    (line_c,) = ax.plot([], [], color="#fff176", lw=2, ls="--", label="Piste estimée fille")
    scat_det = ax.scatter([], [], s=18, c="#b0bec5", alpha=0.7, label="Détections scan")
    gt_line = ax.axvline(x=0, color="#ef5350", ls=":", lw=0)  # placeholder
    det_line = ax.axvline(x=0, color="#66bb6a", ls=":", lw=0)

    title = ax.set_title("", color="white", fontsize=14, fontweight="bold")
    hud = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="#eceff1",
        fontsize=11,
        family="monospace",
    )
    banner = ax.text(
        0.5,
        1.02,
        "",
        transform=ax.transAxes,
        ha="center",
        color="#ffeb3b",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Est (m)", color="#cfd8dc")
    ax.set_ylabel("Nord (m)", color="#cfd8dc")
    ax.tick_params(colors="#90a4ae")
    for spine in ax.spines.values():
        spine.set_color("#37474f")
    ax.legend(loc="upper right", facecolor="#263238", edgecolor="#546e7a", labelcolor="white")
    marker_artists: list = []

    def update(i: int):
        nonlocal marker_artists
        for art in marker_artists:
            art.remove()
        marker_artists = []
        fr = frames[i]
        for tid in ("AC", "MS"):
            if tid in fr.truth_xy:
                trails[tid].append(fr.truth_xy[tid])

        line_t1.set_data([p[0] for p in trails["AC"]], [p[1] for p in trails["AC"]])
        line_t2.set_data([p[0] for p in trails["MS"]], [p[1] for p in trails["MS"]])

        keys = sorted(fr.track_xy.keys())
        p_id = keys[0] if keys else None
        c_id = keys[1] if len(keys) > 1 else None

        if p_id and p_id in fr.track_xy:
            parent_trail.append(fr.track_xy[p_id])
        if c_id and c_id in fr.track_xy:
            child_trail.append(fr.track_xy[c_id])

        line_p.set_data([p[0] for p in parent_trail], [p[1] for p in parent_trail])
        line_c.set_data([p[0] for p in child_trail], [p[1] for p in child_trail])

        if fr.det_xy:
            scat_det.set_offsets(np.array(fr.det_xy))
        else:
            scat_det.set_offsets(np.empty((0, 2)))

        delay_s = None
        if gt_scan is not None and det_scan is not None and i >= det_scan:
            delay_s = det_scan - gt_scan

        title.set_text(
            f"Scénario détachement — scan {fr.scan_index}/{len(frames)-1}  |  t = {fr.timestamp_str}  |  "
            f"{fr.n_dets} détection(s)"
        )

        hud_txt = [
            f"Séparation vérité terrain : scan {gt_scan} (t = 60 s)" if gt_scan is not None else "",
        ]
        if det_scan is not None:
            hud_txt.append(f"Détection confirmée     : scan {det_scan} (Δt = {det_scan - gt_scan}s)" if gt_scan is not None else "")
        else:
            hud_txt.append("Détection confirmée     : —")
        hud.set_text("\n".join(x for x in hud_txt if x))

        msg = ""
        if gt_scan is not None and i == gt_scan:
            msg = "◆ SÉPARATION PHYSIQUE (simulation)"
        elif det_scan is not None and i == det_scan:
            msg = "◆ DÉTECTION CONFIRMÉE (tracker)"
        banner.set_text(msg)

        ac_key = "AC" if "AC" in fr.truth_xy else next(iter(fr.truth_xy.keys()), None)
        if gt_scan is not None and i == gt_scan and ac_key:
            marker_artists.append(
                ax.scatter(
                    [fr.truth_xy[ac_key][0]],
                    [fr.truth_xy[ac_key][1]],
                    s=400,
                    facecolors="none",
                    edgecolors="#ef5350",
                    linewidths=3,
                    zorder=10,
                )
            )
            marker_artists.append(
                ax.text(
                    fr.truth_xy[ac_key][0],
                    fr.truth_xy[ac_key][1] + 800,
                    "Lâcher\nmissile",
                    color="#ef5350",
                    ha="center",
                    fontweight="bold",
                )
            )
        if det_scan is not None and i == det_scan and parent_trail:
            px, py = parent_trail[-1]
            marker_artists.append(
                ax.scatter(
                    [px],
                    [py],
                    s=400,
                    facecolors="none",
                    edgecolors="#66bb6a",
                    linewidths=3,
                    zorder=10,
                )
            )
            marker_artists.append(
                ax.text(
                    px,
                    py - 800,
                    "Confirmation\ndétection",
                    color="#66bb6a",
                    ha="center",
                    fontweight="bold",
                )
            )

        artists = [line_t1, line_t2, line_p, line_c, scat_det, title, hud, banner]
        return artists

    writer = FFMpegWriter(fps=fps, bitrate=4000)
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output), writer=writer, dpi=120)
    plt.close(fig)
    meta_path = output.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "config" / "demo.yaml"))
    p.add_argument("--scenario", default="probable_detach_opposite")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=str(ROOT / "output" / "simulation_film.mp4"))
    p.add_argument("--fps", type=float, default=4.0, help="Images/s (4 = 120s sim en ~30s de vidéo)")
    args = p.parse_args()
    cfg = load_config(args.config)
    frames, meta, _ = build_frames(cfg, args.scenario, args.seed)
    render_mp4(frames, meta, Path(args.output), fps=args.fps)
    print(f"Film: {args.output}")
    if meta.get("delay_s") is not None:
        print(
            f"Séparation GT scan {meta.get('gt_separation_scan')} → "
            f"détection scan {meta.get('detection_scan')} (Δt = {meta['delay_s']} s)"
        )


if __name__ == "__main__":
    main()
