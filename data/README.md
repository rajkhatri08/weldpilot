# data/

Datasets live here on your machine only. Git ignores this folder except this file and `annotations/`.

- `raw/`: downloads, exactly as published. Never edit anything here. Aluminium 5083 goes in `raw/tig-al5083/`.
- `derived/`: files rebuilt by `weldpilot` commands, such as the frame, segment and session indexes (`frames.parquet`, `segments.parquet`, `sessions.parquet`) written by `weldpilot data index`.
  - `phash.parquet`: the perceptual hash of every frame, a cache that `weldpilot audit duplicates` writes. It holds hashes of test frames, which are for the leakage audit only (data rule 2): never use them as features.
- `annotations/`: labels Raj makes by hand. These are committed.
  - `onset_marks.csv`: fault onsets marked by eye, blind to the labels. `weldpilot onset pick` starts it; `weldpilot onset sheets` draws the sheets to mark from.
