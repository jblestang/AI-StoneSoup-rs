use stonesoup_sim::{generate_lot1_tutorial, LOT1_NUMPY_SEED, LOT1_NUM_STEPS};

#[test]
fn lot1_measurements_match_embedded_fixtures() {
    let data = generate_lot1_tutorial(LOT1_NUMPY_SEED).expect("generate");
    assert_eq!(data.measurements.len(), LOT1_NUM_STEPS);
    for i in 0..LOT1_NUM_STEPS {
        let m = fixture_row(&stonesoup_scenarios::MEASUREMENTS, i).expect("row");
        for j in 0..2 {
            let d = (data.measurements[i][j] - m[j]).abs();
            assert!(d <= 1e-9, "meas[{i}][{j}]");
        }
    }
}

#[test]
fn lot1_truth_matches_embedded_fixtures() {
    let data = generate_lot1_tutorial(LOT1_NUMPY_SEED).expect("generate");
    for i in 0..LOT1_NUM_STEPS {
        let t = fixture_row(&stonesoup_scenarios::TRUTH, i).expect("row");
        for j in 0..4 {
            let d = (data.truth[i][j] - t[j]).abs();
            assert!(d <= 1e-9, "truth[{i}][{j}]");
        }
    }
}

fn fixture_row<const N: usize>(table: &[[f64; N]; LOT1_NUM_STEPS], i: usize) -> Option<[f64; N]> {
    table.get(i).copied()
}
