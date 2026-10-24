# data/

Datasets live here on your machine only. Git ignores this folder except this file and `annotations/`.

- `raw/`: downloads, exactly as published. Never edit anything here. Aluminium 5083 goes in `raw/tig-al5083/`.
- `derived/`: files rebuilt by `weldpilot` commands, such as the frame, segment and session indexes (`frames.parquet`, `segments.parquet`, `sessions.parquet`) written by `weldpilot data index`.
- `annotations/`: labels Raj makes by hand. These are committed.
