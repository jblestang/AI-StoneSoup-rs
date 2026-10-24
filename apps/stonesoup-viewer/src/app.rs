use std::f64::consts::TAU;

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints, Points, Polygon};
use stonesoup_scenarios::{Lot1KalmanScenario, Lot1StepOutcome, ScenarioId, ScenarioRegistry};

/// NumPy / Stone Soup fixture seed (see `tools/python/export_lot1_fixtures.py`).
const LOT1_FIXTURE_SEED: u32 = 1991;

#[derive(Default)]
pub struct ViewerApp {
    lot1: Option<Lot1KalmanScenario>,
    playing: bool,
    speed: f32,
    error: Option<String>,
    export_notice: Option<String>,
    truth_trail: Vec<[f64; 2]>,
    meas_trail: Vec<[f64; 2]>,
    est_trail: Vec<[f64; 2]>,
    last_ellipse: Option<EllipseOverlay>,
    last_advance: f64,
}

#[derive(Clone, Copy)]
struct EllipseOverlay {
    center: [f64; 2],
    semi: (f64, f64),
    angle_rad: f64,
}

fn ellipse_ring(
    center: [f64; 2],
    semi: (f64, f64),
    angle_rad: f64,
    segments: usize,
) -> Vec<[f64; 2]> {
    let (a, b) = semi;
    let (sin_t, cos_t) = angle_rad.sin_cos();
    (0..segments)
        .map(|i| {
            let phi = TAU * (i as f64) / (segments as f64);
            let (sin_p, cos_p) = phi.sin_cos();
            let lx = a * cos_p;
            let ly = b * sin_p;
            [
                center[0] + lx * cos_t - ly * sin_t,
                center[1] + lx * sin_t + ly * cos_t,
            ]
        })
        .collect()
}

impl ViewerApp {
    fn ensure_scenario(&mut self) {
        if self.lot1.is_none() {
            match ScenarioRegistry::spawn(ScenarioId::Lot1KalmanTutorial) {
                Ok(s) => self.lot1 = Some(s),
                Err(e) => self.error = Some(format!("{e:?}")),
            }
        }
    }

    pub fn reset(&mut self) {
        self.truth_trail.clear();
        self.meas_trail.clear();
        self.est_trail.clear();
        self.last_ellipse = None;
        self.playing = false;
        self.last_advance = 0.0;
        self.export_notice = None;
        if let Some(s) = &mut self.lot1 {
            if let Err(e) = s.reset() {
                self.error = Some(format!("{e:?}"));
            }
        }
    }

    pub fn step_once(&mut self) {
        let Some(s) = &mut self.lot1 else { return };
        match s.step() {
            Ok(Lot1StepOutcome::Stepped | Lot1StepOutcome::Finished) => {
                if let Ok(snap) = s.snapshot() {
                    self.truth_trail.push([snap.truth_xy.0, snap.truth_xy.1]);
                    self.meas_trail
                        .push([snap.measurement_xy.0, snap.measurement_xy.1]);
                    self.est_trail
                        .push([snap.estimate_xy.0, snap.estimate_xy.1]);
                    self.last_ellipse = Some(EllipseOverlay {
                        center: [snap.estimate_xy.0, snap.estimate_xy.1],
                        semi: snap.ellipse_semi,
                        angle_rad: snap.ellipse_angle_rad,
                    });
                }
            }
            Err(e) => self.error = Some(format!("{e:?}")),
        }
    }

    fn export_json(&mut self) -> Option<String> {
        let s = self.lot1.as_ref()?;
        let snap = s.snapshot().ok()?;
        let payload = serde_json::json!({
            "scenario_id": "Lot1KalmanTutorial",
            "upstream": "docs/tutorials/01_KalmanFilterTutorial.py",
            "fixture_seed": LOT1_FIXTURE_SEED,
            "step": snap.step,
            "max_steps": stonesoup_scenarios::MAX_LOT1_STEPS,
            "rmse": snap.rmse,
            "finished": snap.finished,
            "truth_trail": self.truth_trail,
            "measurement_trail": self.meas_trail,
            "estimate_trail": self.est_trail,
            "last_ellipse": {
                "semi_axes": [snap.ellipse_semi.0, snap.ellipse_semi.1],
                "angle_rad": snap.ellipse_angle_rad,
                "center": [snap.estimate_xy.0, snap.estimate_xy.1],
            },
        });
        serde_json::to_string_pretty(&payload).ok()
    }
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.ensure_scenario();
        if self.playing {
            let now = ctx.input(|i| i.time);
            let delay = 0.4 / f64::from(self.speed.max(0.1));
            if now - self.last_advance >= delay {
                self.step_once();
                self.last_advance = now;
            }
            ctx.request_repaint();
        }

        egui::SidePanel::left("catalog").show(ctx, |ui| {
            ui.heading("Catalogue");
            for desc in ScenarioRegistry::all() {
                ui.separator();
                ui.label(desc.title);
                ui.label(format!("Upstream: {}", desc.upstream_path));
                ui.label(format!("SHA: {}", &desc.upstream_sha[..12]));
            }
            ui.separator();
            ui.label("Lot 1 — Kalman tutorial (active)");
            ui.label(format!("Fixture seed: {LOT1_FIXTURE_SEED}"));
            if ui.button("Reset").clicked() {
                self.reset();
            }
            if ui
                .button(if self.playing { "Pause" } else { "Play" })
                .clicked()
            {
                self.playing = !self.playing;
            }
            if ui.button("Step").clicked() {
                self.step_once();
            }
            ui.add(egui::Slider::new(&mut self.speed, 0.1..=4.0).text("speed"));
            if ui.button("Export results JSON").clicked() {
                if let Some(json) = self.export_json() {
                    ui.output_mut(|o| o.copied_text = json.clone());
                    self.export_notice = Some("Copied JSON to clipboard".into());
                } else {
                    self.export_notice = Some("Export failed".into());
                }
            }
            if let Some(msg) = &self.export_notice {
                ui.label(msg);
            }
            if let Some(s) = &self.lot1 {
                if let Ok(snap) = s.snapshot() {
                    ui.label(format!(
                        "Step {} / {}",
                        snap.step,
                        stonesoup_scenarios::MAX_LOT1_STEPS
                    ));
                    ui.label(format!("RMSE: {:.4}", snap.rmse));
                    ui.label(format!(
                        "Ellipse semi-axes: {:.2}, {:.2}",
                        snap.ellipse_semi.0, snap.ellipse_semi.1
                    ));
                    if snap.finished {
                        ui.colored_label(egui::Color32::LIGHT_GREEN, "Finished");
                    }
                }
            }
            if let Some(err) = &self.error {
                ui.colored_label(egui::Color32::RED, err);
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Tracking view (x–y)");
            let truth_pts: PlotPoints = self.truth_trail.iter().map(|p| [p[0], p[1]]).collect();
            let meas_pts: PlotPoints = self.meas_trail.iter().map(|p| [p[0], p[1]]).collect();
            let est_pts: PlotPoints = self.est_trail.iter().map(|p| [p[0], p[1]]).collect();
            Plot::new("lot1_plot")
                .legend(Legend::default())
                .data_aspect(1.0)
                .show(ui, |plot_ui| {
                    if !self.truth_trail.is_empty() {
                        plot_ui.line(
                            Line::new(truth_pts)
                                .name("Ground truth")
                                .color(egui::Color32::LIGHT_GRAY),
                        );
                    }
                    if !self.meas_trail.is_empty() {
                        plot_ui.points(
                            Points::new(meas_pts)
                                .name("Detections")
                                .radius(4.0)
                                .color(egui::Color32::LIGHT_BLUE),
                        );
                    }
                    if !self.est_trail.is_empty() {
                        plot_ui.line(Line::new(est_pts).name("Track").color(egui::Color32::GREEN));
                    }
                    if let Some(el) = self.last_ellipse {
                        let ring = ellipse_ring(el.center, el.semi, el.angle_rad, 48);
                        let ring_pts: PlotPoints = ring.iter().map(|p| [p[0], p[1]]).collect();
                        plot_ui.polygon(
                            Polygon::new(ring_pts)
                                .name("Covariance (2σ)")
                                .stroke(egui::Stroke::new(
                                    1.0,
                                    egui::Color32::from_rgb(80, 200, 120),
                                ))
                                .fill_color(egui::Color32::from_rgba_unmultiplied(
                                    80, 200, 120, 40,
                                )),
                        );
                    }
                });
        });
    }
}
