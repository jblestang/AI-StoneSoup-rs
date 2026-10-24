use std::f64::consts::TAU;

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints, Points, Polygon};
use stonesoup_scenarios::{
    lot5_case_name, AnyScenario, Scenario, ScenarioId, ScenarioRegistry, ScenarioStepOutcome,
    ViewerSnapshot, TUTORIAL02_SENSOR_XY,
};

/// NumPy / Stone Soup fixture seed (tutorials 01–10).
const FIXTURE_SEED: u32 = 1991;

/// Document title set when `?finish=1` run completes (WASM screenshot automation).
#[cfg(target_arch = "wasm32")]
pub const WASM_READY_TITLE: &str = "stonesoup-viewer-ready";
#[cfg(target_arch = "wasm32")]
pub const WASM_RESET_TITLE: &str = "stonesoup-viewer-reset";
#[cfg(target_arch = "wasm32")]
pub const WASM_STEP_TITLE_PREFIX: &str = "stonesoup-viewer-step-";

#[cfg(target_arch = "wasm32")]
fn set_wasm_document_title(title: &str) {
    if let Some(window) = web_sys::window() {
        let _ = window.document().and_then(|d| {
            d.set_title(title);
            Some(())
        });
    }
}

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
    truth_b_trail: Vec<[f64; 2]>,
    est_b_trail: Vec<[f64; 2]>,
    last_ellipse: Option<EllipseOverlay>,
    last_advance: f64,
    /// Current-step detections (Tutorial 05 clutter).
    step_clutter: Vec<[f64; 2]>,
    step_true_dets: Vec<[f64; 2]>,
    step_associated: Option<[f64; 2]>,
    /// Lot 5 resampling: `(particle index, state value)` at the current step.
    resample_plot: Vec<[f64; 2]>,
    #[cfg(target_arch = "wasm32")]
    wasm_query_handled: bool,
    #[cfg(target_arch = "wasm32")]
    pending_auto_finish: bool,
    #[cfg(target_arch = "wasm32")]
    pending_partial_steps: usize,
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
            truth_b_trail: Vec::new(),
            est_b_trail: Vec::new(),
            last_ellipse: None,
            last_advance: 0.0,
            step_clutter: Vec::new(),
            step_true_dets: Vec::new(),
            step_associated: None,
            resample_plot: Vec::new(),
            #[cfg(target_arch = "wasm32")]
            wasm_query_handled: false,
            #[cfg(target_arch = "wasm32")]
            pending_auto_finish: false,
            #[cfg(target_arch = "wasm32")]
            pending_partial_steps: 0,
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn scenario_id_from_tutorial(code: &str) -> Option<ScenarioId> {
    match code {
        "01" | "1" => Some(ScenarioId::Lot1KalmanTutorial),
        "02" | "2" => Some(ScenarioId::Lot2EkfTutorial),
        "03" | "3" => Some(ScenarioId::Lot2UkfTutorial),
        "04" | "4" => Some(ScenarioId::Lot2ParticleTutorial),
        "05" | "5" => Some(ScenarioId::Lot3ClutterTutorial),
        "06" | "6" => Some(ScenarioId::Lot3MultiTargetTutorial),
        "07" | "7" => Some(ScenarioId::Lot3PdaTutorial),
        "08" | "8" => Some(ScenarioId::Lot3JpdaTutorial),
        "09" | "9" => Some(ScenarioId::Lot4InitiatorsTutorial),
        "10" => Some(ScenarioId::Lot4SimulationTutorial),
        "11" | "resampling" | "sample" => Some(ScenarioId::Lot5ResamplingTutorial),
        "12" | "information" | "if" => Some(ScenarioId::Lot6InformationFilterTutorial),
        "13" | "gmphd" | "phd" | "gm-phd" => Some(ScenarioId::Lot7GmphdTutorial),
        _ => None,
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
        self.truth_b_trail.clear();
        self.est_b_trail.clear();
        self.last_ellipse = None;
        self.step_clutter.clear();
        self.step_true_dets.clear();
        self.step_associated = None;
        self.resample_plot.clear();
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
        #[cfg(target_arch = "wasm32")]
        set_wasm_document_title(WASM_RESET_TITLE);
    }

    fn apply_snapshot(&mut self, snap: ViewerSnapshot) {
        if snap.resample_case != u8::MAX {
            self.resample_plot.clear();
            for i in 0..snap.resample_nparts as usize {
                self.resample_plot.push(snap.resample_xy[i]);
            }
            return;
        }
        self.truth_trail.push([snap.truth_xy.0, snap.truth_xy.1]);
        self.step_clutter.clear();
        self.step_true_dets.clear();
        for i in 0..snap.det_count as usize {
            let p = [snap.det_xy[i][0], snap.det_xy[i][1]];
            if snap.det_is_true[i] {
                self.step_true_dets.push(p);
            } else {
                self.step_clutter.push(p);
            }
        }
        self.step_associated = snap.associated_xy.map(|(x, y)| [x, y]);
        if let Some(p) = self.step_associated {
            self.meas_trail.push(p);
        } else if snap.det_count == 0 {
            self.meas_trail
                .push([snap.measurement_xy.0, snap.measurement_xy.1]);
        }
        self.est_trail
            .push([snap.estimate_xy.0, snap.estimate_xy.1]);
        if let Some(t) = snap.truth_b_xy {
            self.truth_b_trail.push([t.0, t.1]);
        }
        if let Some(e) = snap.estimate_b_xy {
            self.est_b_trail.push([e.0, e.1]);
        }
        self.last_ellipse = snap.ellipse_semi.map(|semi| EllipseOverlay {
            center: [snap.estimate_xy.0, snap.estimate_xy.1],
            semi,
            angle_rad: snap.ellipse_angle_rad,
        });
    }

    /// Advance until the scenario reports finished; rebuild trails for the plot.
    pub fn finish_to_end(&mut self) {
        self.playing = false;
        self.clear_trails();
        self.ensure_scenario();
        loop {
            let progress = {
                let Some(s) = &mut self.scenario else {
                    break;
                };
                match s.step() {
                    Ok(ScenarioStepOutcome::Stepped) => (true, s.viewer_snapshot().ok()),
                    Ok(ScenarioStepOutcome::Finished) => (false, s.viewer_snapshot().ok()),
                    Err(e) => {
                        self.error = Some(format!("{e:?}"));
                        break;
                    }
                }
            };
            if let Some(snap) = progress.1 {
                self.apply_snapshot(snap);
            }
            if !progress.0 {
                break;
            }
        }
        #[cfg(target_arch = "wasm32")]
        set_wasm_document_title(WASM_READY_TITLE);
    }

    #[cfg(target_arch = "wasm32")]
    fn apply_wasm_query_params(&mut self) {
        if self.wasm_query_handled {
            return;
        }
        self.wasm_query_handled = true;
        let Some(window) = web_sys::window() else {
            return;
        };
        let Ok(search) = window.location().search() else {
            return;
        };
        let mut tutorial: Option<String> = None;
        let mut finish = false;
        let mut reset = false;
        let mut partial_steps: Option<usize> = None;
        for pair in search.trim_start_matches('?').split('&') {
            let Some((key, value)) = pair.split_once('=') else {
                continue;
            };
            match key {
                "tutorial" => tutorial = Some(value.to_string()),
                "finish" if value == "1" || value.eq_ignore_ascii_case("true") => finish = true,
                "reset" if value == "1" || value.eq_ignore_ascii_case("true") => reset = true,
                "steps" => partial_steps = value.parse().ok(),
                _ => {}
            }
        }
        if let Some(code) = tutorial.as_deref().and_then(scenario_id_from_tutorial) {
            self.load_scenario(code);
        }
        if reset {
            self.reset();
        }
        if finish {
            self.pending_auto_finish = true;
        } else if let Some(n) = partial_steps {
            self.pending_partial_steps = n;
        }
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
        matches!(
            self.active_id,
            ScenarioId::Lot2EkfTutorial
                | ScenarioId::Lot2UkfTutorial
                | ScenarioId::Lot2ParticleTutorial
        )
    }

    fn is_lot3_meas(&self) -> bool {
        matches!(
            self.active_id,
            ScenarioId::Lot3ClutterTutorial
                | ScenarioId::Lot3MultiTargetTutorial
                | ScenarioId::Lot3PdaTutorial
                | ScenarioId::Lot3JpdaTutorial
                | ScenarioId::Lot4InitiatorsTutorial
                | ScenarioId::Lot4SimulationTutorial
        )
    }

    fn is_lot3_nn(&self) -> bool {
        matches!(self.active_id, ScenarioId::Lot3ClutterTutorial)
    }

    fn is_lot5_resampling(&self) -> bool {
        matches!(self.active_id, ScenarioId::Lot5ResamplingTutorial)
    }
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        #[cfg(target_arch = "wasm32")]
        self.apply_wasm_query_params();
        self.ensure_scenario();
        #[cfg(target_arch = "wasm32")]
        if self.pending_auto_finish {
            self.pending_auto_finish = false;
            self.finish_to_end();
        }
        #[cfg(target_arch = "wasm32")]
        if self.pending_partial_steps > 0 {
            let n = self.pending_partial_steps;
            self.pending_partial_steps = 0;
            for _ in 0..n {
                self.step_once();
            }
            let step = self
                .scenario
                .as_ref()
                .and_then(|s| s.viewer_snapshot().ok())
                .map(|snap| snap.step)
                .unwrap_or(0);
            set_wasm_document_title(&format!("{WASM_STEP_TITLE_PREFIX}{step}"));
        }
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
                    if self.is_lot5_resampling() && snap.resample_case != u8::MAX {
                        ui.label(format!("Resampler: {}", lot5_case_name(snap.resample_case)));
                        ui.label(format!("Particles: {}", snap.resample_nparts));
                    } else if !self.is_lot5_resampling() {
                        ui.label(format!("RMSE: {:.4}", snap.rmse));
                    }
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
            if self.is_lot5_resampling() {
                ui.heading("Resampled particles (index → state)");
                Plot::new("resample_plot")
                    .legend(Legend::default())
                    .data_aspect(1.0)
                    .show(ui, |plot_ui| {
                        if !self.resample_plot.is_empty() {
                            let pts: PlotPoints =
                                self.resample_plot.iter().map(|p| [p[0], p[1]]).collect();
                            plot_ui.points(
                                Points::new(pts)
                                    .name("Particles")
                                    .radius(6.0)
                                    .color(egui::Color32::from_rgb(120, 180, 255)),
                            );
                        }
                    });
                return;
            }
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
                    if self.is_lot3_meas() && !self.step_clutter.is_empty() {
                        let clutter: PlotPoints =
                            self.step_clutter.iter().map(|p| [p[0], p[1]]).collect();
                        plot_ui.points(
                            Points::new(clutter)
                                .name("Clutter")
                                .radius(3.0)
                                .color(egui::Color32::from_rgb(180, 120, 80)),
                        );
                    }
                    if self.is_lot3_meas() && !self.step_true_dets.is_empty() {
                        let truth_dets: PlotPoints =
                            self.step_true_dets.iter().map(|p| [p[0], p[1]]).collect();
                        plot_ui.points(
                            Points::new(truth_dets)
                                .name("True detections (step)")
                                .radius(5.0)
                                .color(egui::Color32::LIGHT_BLUE),
                        );
                    }
                    if !self.is_lot3_meas() && !self.meas_trail.is_empty() {
                        plot_ui.points(
                            Points::new(meas_pts)
                                .name("Detections (Cartesian)")
                                .radius(4.0)
                                .color(egui::Color32::LIGHT_BLUE),
                        );
                    }
                    if self.is_lot3_nn() {
                        if let Some(p) = self.step_associated {
                            plot_ui.points(
                                Points::new(vec![[p[0], p[1]]])
                                    .name("Associated detection")
                                    .radius(6.0)
                                    .color(egui::Color32::from_rgb(255, 200, 60)),
                            );
                        }
                    }
                    if !self.est_trail.is_empty() {
                        plot_ui.line(
                            Line::new(est_pts)
                                .name("Track A")
                                .color(egui::Color32::GREEN),
                        );
                    }
                    if !self.truth_b_trail.is_empty() {
                        let tb: PlotPoints =
                            self.truth_b_trail.iter().map(|p| [p[0], p[1]]).collect();
                        plot_ui.line(
                            Line::new(tb)
                                .name("Ground truth B")
                                .color(egui::Color32::from_rgb(200, 200, 255)),
                        );
                    }
                    if !self.est_b_trail.is_empty() {
                        let eb: PlotPoints =
                            self.est_b_trail.iter().map(|p| [p[0], p[1]]).collect();
                        plot_ui.line(
                            Line::new(eb)
                                .name("Track B")
                                .color(egui::Color32::from_rgb(255, 140, 80)),
                        );
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
