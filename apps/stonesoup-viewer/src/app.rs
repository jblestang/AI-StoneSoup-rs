use std::f64::consts::TAU;

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints, Points, Polygon};
use stonesoup_scenarios::{
    AnyScenario, Scenario, ScenarioId, ScenarioRegistry, ScenarioStepOutcome, ViewerSnapshot,
    TUTORIAL02_SENSOR_XY,
};

/// NumPy / Stone Soup fixture seed (tutorials 01–04).
const FIXTURE_SEED: u32 = 1991;

pub struct ViewerApp {
    active_id: ScenarioId,
    scenario: Option<AnyScenario>,
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

impl Default for ViewerApp {
    fn default() -> Self {
        Self {
            active_id: ScenarioId::Lot1KalmanTutorial,
            scenario: None,
            playing: false,
            speed: 1.0,
            error: None,
            export_notice: None,
            truth_trail: Vec::new(),
            meas_trail: Vec::new(),
            est_trail: Vec::new(),
            last_ellipse: None,
            last_advance: 0.0,
        }
    }
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
    fn load_scenario(&mut self, id: ScenarioId) {
        match AnyScenario::spawn(id) {
            Ok(s) => {
                self.active_id = id;
                self.scenario = Some(s);
                self.error = None;
                self.clear_trails();
            }
            Err(e) => self.error = Some(format!("{e:?}")),
        }
    }

    fn ensure_scenario(&mut self) {
        if self.scenario.is_none() {
            self.load_scenario(self.active_id);
        }
    }

    fn clear_trails(&mut self) {
        self.truth_trail.clear();
        self.meas_trail.clear();
        self.est_trail.clear();
        self.last_ellipse = None;
    }

    pub fn reset(&mut self) {
        self.clear_trails();
        self.playing = false;
        self.last_advance = 0.0;
        self.export_notice = None;
        if let Some(s) = &mut self.scenario {
            if let Err(e) = s.reset() {
                self.error = Some(format!("{e:?}"));
            }
        }
    }

    fn apply_snapshot(&mut self, snap: ViewerSnapshot) {
        self.truth_trail.push([snap.truth_xy.0, snap.truth_xy.1]);
        self.meas_trail
            .push([snap.measurement_xy.0, snap.measurement_xy.1]);
        self.est_trail
            .push([snap.estimate_xy.0, snap.estimate_xy.1]);
        self.last_ellipse = snap.ellipse_semi.map(|semi| EllipseOverlay {
            center: [snap.estimate_xy.0, snap.estimate_xy.1],
            semi,
            angle_rad: snap.ellipse_angle_rad,
        });
    }

    pub fn step_once(&mut self) {
        let Some(s) = &mut self.scenario else {
            return;
        };
        match s.step() {
            Ok(ScenarioStepOutcome::Stepped | ScenarioStepOutcome::Finished) => {
                if let Ok(snap) = s.viewer_snapshot() {
                    self.apply_snapshot(snap);
                }
            }
            Err(e) => self.error = Some(format!("{e:?}")),
        }
    }

    fn export_json(&mut self) -> Option<String> {
        let s = self.scenario.as_ref()?;
        let snap = s.viewer_snapshot().ok()?;
        let desc = ScenarioRegistry::all()
            .iter()
            .find(|d| d.id == self.active_id)?;
        let payload = serde_json::json!({
            "scenario_id": format!("{:?}", self.active_id),
            "upstream": desc.upstream_path,
            "fixture_seed": FIXTURE_SEED,
            "step": snap.step,
            "max_steps": snap.max_steps,
            "rmse": snap.rmse,
            "finished": snap.finished,
            "truth_trail": self.truth_trail,
            "measurement_trail": self.meas_trail,
            "estimate_trail": self.est_trail,
            "bearing_range_last": snap.bearing_range,
            "last_ellipse": snap.ellipse_semi.map(|semi| serde_json::json!({
                "semi_axes": [semi.0, semi.1],
                "angle_rad": snap.ellipse_angle_rad,
                "center": [snap.estimate_xy.0, snap.estimate_xy.1],
            })),
        });
        serde_json::to_string_pretty(&payload).ok()
    }

    fn is_lot2(&self) -> bool {
        !matches!(self.active_id, ScenarioId::Lot1KalmanTutorial)
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
                let selected = self.active_id == desc.id;
                if ui
                    .selectable_label(selected, if selected { "Active" } else { "Load" })
                    .clicked()
                    && !selected
                {
                    self.playing = false;
                    self.load_scenario(desc.id);
                }
            }
            ui.separator();
            ui.label(format!("Fixture seed: {FIXTURE_SEED}"));
            if self.is_lot2() {
                ui.label(format!(
                    "Sensor @ ({:.0}, {:.0}) m — bearing–range",
                    TUTORIAL02_SENSOR_XY.0, TUTORIAL02_SENSOR_XY.1
                ));
            }
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
            if let Some(s) = &self.scenario {
                if let Ok(snap) = s.viewer_snapshot() {
                    ui.label(format!("Step {} / {}", snap.step, snap.max_steps));
                    ui.label(format!("RMSE: {:.4}", snap.rmse));
                    if let Some((b, r)) = snap.bearing_range {
                        ui.label(format!(
                            "Last detection: bearing {b:.2} rad, range {r:.2} m"
                        ));
                    }
                    if let Some(semi) = snap.ellipse_semi {
                        ui.label(format!("Ellipse semi-axes: {:.2}, {:.2}", semi.0, semi.1));
                    } else if matches!(self.active_id, ScenarioId::Lot2ParticleTutorial) {
                        ui.label("Particle mean (no covariance ellipse)");
                    }
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
            let sensor = if self.is_lot2() {
                Some(TUTORIAL02_SENSOR_XY)
            } else {
                None
            };
            Plot::new("tracking_plot")
                .legend(Legend::default())
                .data_aspect(1.0)
                .show(ui, |plot_ui| {
                    if let Some((sx, sy)) = sensor {
                        plot_ui.points(
                            Points::new(vec![[sx, sy]])
                                .name("Sensor")
                                .radius(6.0)
                                .color(egui::Color32::YELLOW),
                        );
                    }
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
                                .name("Detections (Cartesian)")
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
