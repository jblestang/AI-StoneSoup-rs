#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use stonesoup_viewer::ViewerApp;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([1024.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Stone Soup Rust — viewer",
        native_options,
        Box::new(|_| Ok(Box::<ViewerApp>::default())),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast;
    let web_options = eframe::WebOptions::default();
    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("window")
            .document()
            .expect("document");
        let canvas = document
            .get_element_by_id("stonesoup_canvas")
            .expect("canvas")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("canvas element");
        let start_result = eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|_| Ok(Box::<ViewerApp>::default())),
            )
            .await;
        if let Some(err) = start_result.err() {
            web_sys::console::error_1(&err);
        }
    });
}
