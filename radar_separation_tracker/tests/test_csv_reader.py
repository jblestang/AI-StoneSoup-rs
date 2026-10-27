from datetime import datetime

from radar_tracker.csv_reader import DetectionRecord, group_by_scan, read_detections_csv


def test_group_by_scan(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(
        "timestamp,radar_id,scan_id,detection_id,range_m,azimuth_rad,elevation_rad\n"
        "2026-01-01T12:00:00,R1,S1,D1,1000,0.1,0.2\n"
        "2026-01-01T12:00:00,R1,S1,D2,1001,0.1,0.2\n"
        "2026-01-01T12:00:01,R1,S2,D1,1002,0.1,0.2\n"
    )
    recs = read_detections_csv(p)
    batches = group_by_scan(recs)
    assert len(batches) == 2
    assert len(batches[0].detections) == 2
