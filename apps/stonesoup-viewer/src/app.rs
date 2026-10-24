use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints, Points};
use stonesoup_scenarios::{Lot1KalmanScenario, Lot1StepOutcome, ScenarioId, ScenarioRegistry};

#[derive(Default)]
pub struct ViewerApp {
    lot1: Option<Lot1KalmanScenario>,
    playing: bool,
    speed: f32,
    error: Option<String>,
    truth_trail: Vec<[f64; 2]>,
    meas_trail: Vec<[f64; 2]>,
    est_trail: Vec<[f64; 2]>,
    last_advance: f64,
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
        self.playing = false;
        self.last_advance = 0.0;
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
                }
            }
            Err(e) => self.error = Some(format!("{e:?}")),
        }
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
            if let Some(s) = &self.lot1 {
                if let Ok(snap) = s.snapshot() {
                    ui.label(format!(
                        "Step {} / {}",
                        snap.step,
                        stonesoup_scenarios::MAX_LOT1_STEPS
                    ));
                    ui.label(format!("RMSE: {:.4}", snap.rmse));
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
                });
        });
    }
}
