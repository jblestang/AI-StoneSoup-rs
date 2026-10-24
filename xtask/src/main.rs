use std::process::{Command, ExitCode};

fn run(mut cmd: Command) -> bool {
    eprintln!("+ {:?}", cmd);
    match cmd.status() {
        Ok(s) if s.success() => true,
        Ok(s) => {
            eprintln!("exit {}", s);
            false
        }
        Err(e) => {
            eprintln!("{e}");
            false
        }
    }
}

fn cargo(sub: &str, args: &[&str]) -> Command {
    let mut c = Command::new("cargo");
    c.arg(sub).args(args);
    c
}

fn trunk_on_path() -> bool {
    Command::new("bash")
        .args(["-lc", "command -v trunk >/dev/null"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn should_run_wasm_smoke() -> bool {
    if std::env::var("NO_COLOR").is_ok() {
        return false;
    }
    if std::env::var("VERIFY_WASM").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true")) {
        return true;
    }
    if std::env::var("CI").is_ok() {
        return false;
    }
    trunk_on_path()
}

fn main() -> ExitCode {
    let mut ok = true;
    ok &= run(cargo("fmt", &["--all", "--", "--check"]));
    ok &= run(cargo(
        "clippy",
        &[
            "-p",
            "stonesoup-core",
            "-p",
            "stonesoup-sim",
            "-p",
            "stonesoup-scenarios",
            "--no-deps",
            "--locked",
            "--",
            "-D",
            "warnings",
        ],
    ));
    ok &= run(cargo("test", &["--workspace", "--locked"]));
    ok &= run(cargo(
        "check",
        &[
            "-p",
            "stonesoup-core",
            "--no-default-features",
            "--target",
            "thumbv7em-none-eabihf",
            "--locked",
        ],
    ));
    ok &= run(cargo(
        "check",
        &[
            "-p",
            "stonesoup-core",
            "--no-default-features",
            "--target",
            "wasm32-unknown-unknown",
            "--locked",
        ],
    ));
    ok &= run({
        let mut c = Command::new("cargo");
        c.args([
            "run",
            "--manifest-path",
            "verification/no-panic-harness/Cargo.toml",
            "--profile",
            "no-panic-test",
            "--locked",
        ]);
        c
    });
    ok &= run({
        let mut c = Command::new("bash");
        c.arg("verification/scripts/check_catalogue.sh");
        c
    });
    ok &= run({
        let mut c = Command::new("bash");
        c.arg("verification/scripts/mutation_no_panic.sh");
        c
    });
    ok &= run(cargo(
        "run",
        &[
            "-p",
            "stonesoup-headless",
            "--release",
            "--locked",
            "--",
            "lot1",
        ],
    ));
    if should_run_wasm_smoke() {
        ok &= run({
            let mut c = Command::new("bash");
            c.arg("-lc");
            c.arg("cd apps/stonesoup-viewer && unset NO_COLOR && trunk build --release");
            c
        });
        ok &= run({
            let mut c = Command::new("bash");
            c.arg("verification/scripts/wasm_viewer_smoke.sh");
            c
        });
    } else {
        eprintln!("xtask verify: skipping Trunk/WASM smoke (set VERIFY_WASM=1 to force in CI)");
    }
    if ok {
        eprintln!("xtask verify: all executed checks passed");
        ExitCode::SUCCESS
    } else {
        eprintln!("xtask verify: failed");
        ExitCode::FAILURE
    }
}
