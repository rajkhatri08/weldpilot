# Datasheet: TIG aluminium 5083 weld images

Structure follows Gebru et al., "Datasheets for Datasets" (Communications of the ACM, 2021).

## Motivation

## Composition

### By split

| Split | Folders | Sessions | Frames | Footage (s) |
| :--- | ---: | ---: | ---: | ---: |
| train | 46 | 32 | 26,666 | 484.8 |
| test | 14 | 12 | 6,588 | 119.8 |
| all | 60 | 39 | 33,254 | 604.6 |

Sessions: 27 train-only, 7 test-only and 5 crossing (folders in both train/ and test/), 39 in all. A crossing session counts in both split rows. Footage is the frames present divided by 55 fps, an assumption (the camera rate reported for the SS304 set); missing frame numbers add no footage.

Crossing sessions, never used for training, calibration or model selection (data rule 3): 170904-150144, 170904-152301, 170904-154202, 170906-113317, 170906-153326.

### By class

| Class | Frames | Folders | Train-only sessions | Test-only sessions | Crossing sessions | Flag |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| 0 good weld | 10,947 | 16 | 12 | 3 | 1 |  |
| 1 burn through | 2,134 | 3 | 1 | 1 | 1 | fewer than 2 train-only sessions |
| 2 contamination | 8,403 | 16 | 11 | 2 | 3 |  |
| 3 lack of fusion | 5,035 | 6 | 3 | 2 | 1 |  |
| 4 misalignment | 3,682 | 12 | 8 | 1 | 3 |  |
| 5 lack of penetration | 3,053 | 7 | 5 | 0 | 2 |  |

A session can hold several classes, so the session columns don't add up to 39. Flagged (fewer than 2 train-only sessions): 1 burn through. Such a class can't have one session held out for validation while another trains; crossing sessions count for neither (data rule 3).

Source: `weldpilot data composition`, from the index built at git 117864f07990f501b78306dd29bde0cc24376aaf on the dataset with fingerprint 931a5ad08ea12a7271a47fb77d47fd33ea19720c22c3451b5386d9c8627dd4cc (checked against configs/dataset.yaml). Counts are exact for this release, not estimates.

### The al5083/ copy

The Kaggle archive also ships `al5083/`, a byte-identical copy of train/, test/ and README (checked with `diff -rq -x .DS_Store -x al5083 data/raw/tig-al5083 data/raw/tig-al5083/al5083` on 2026-10-24: no output, exit 0). It is ignored.

## Collection process

## Preprocessing, cleaning and labelling

## Uses

## Distribution

## Maintenance
