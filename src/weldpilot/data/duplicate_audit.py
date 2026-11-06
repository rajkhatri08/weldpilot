"""Near-duplicate audit: which pairs of near-identical frames matter, and the report.

`weldpilot audit duplicates` hashes every frame and finds the pairs whose hashes are a few
bits apart (`phash.near_pairs`). This module sorts those pairs into kinds by the sessions and
split folders of their frames, picks the pairs that get a thumbnail, and writes
duplicates.csv and duplicates.md. These functions are pure; the command reads and writes.

Session roles come from `weldpilot.data.splits` (data rule 4): training sessions, and test
sessions, crossing ones included. duplicates.csv never holds labels, and the class table
leaves out every pair with a frame in a session marked blind for onset pass 2.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from weldpilot.data.al5083 import CLASS_NAMES
from weldpilot.data.composition import markdown_table
from weldpilot.data.phash import HASH_BITS, NearPairs

TRAINING_VS_TEST = "training_vs_test"
CROSSING_SAME_SESSION = "crossing_same_session"
CROSSING_VS_OTHER_TEST = "crossing_vs_other_test"
TRAINING_VS_TRAINING = "training_vs_training"
KINDS = (TRAINING_VS_TEST, CROSSING_SAME_SESSION, CROSSING_VS_OTHER_TEST, TRAINING_VS_TRAINING)
"""In report order."""

CLASS_KINDS = (TRAINING_VS_TEST, TRAINING_VS_TRAINING)
"""The kinds that get thumbnails and a class table: both involve frames that can train."""

THUMBNAIL_DIR = "duplicate_thumbnails"
"""Next to duplicates.csv."""

FRAME_COLUMNS = ("session_id", "split", "frame_number")
"""What the pairs need from frames.parquet: no labels."""

LABEL_COLUMNS = ("session_id", "split", "frame_number", "label")
"""What the class table needs, read only for frames outside the marked sessions."""

PAIR_COLUMNS = (
    "kind",
    "distance",
    "a_split",
    "a_session",
    "a_frame",
    "b_split",
    "b_session",
    "b_frame",
)
CSV_COLUMNS = (*PAIR_COLUMNS, "thumbnail")

TOP_SESSION_PAIRS = 5
"""Session pairs listed per kind in duplicates.md."""


def check_frames(frames: pd.DataFrame, training: Collection[str], test: Collection[str]) -> None:
    """Raise ValueError if the frames table disagrees with the session lists.

    Names any missing column, a split other than train or test, sessions that are neither
    training nor test sessions, and training sessions with frames in test/.
    """
    missing = [name for name in FRAME_COLUMNS if name not in frames.columns]
    if missing:
        raise ValueError(f"the frames table is missing columns: {', '.join(missing)}")
    odd = sorted(set(frames["split"].astype(str)) - {"train", "test"})
    if odd:
        raise ValueError(f"splits other than train and test: {', '.join(odd)}")
    unknown = sorted(set(frames["session_id"].astype(str)) - set(training) - set(test))
    if unknown:
        raise ValueError(
            f"sessions that are neither training nor test sessions: {', '.join(unknown)}"
        )
    in_test = set(frames.loc[frames["split"] == "test", "session_id"].astype(str))
    wrong = sorted(in_test & set(training))
    if wrong:
        raise ValueError(f"training sessions with frames in test/: {', '.join(wrong)}")


def _in_report_order(pairs: pd.DataFrame) -> pd.DataFrame:
    rank = pairs["kind"].map({kind: n for n, kind in enumerate(KINDS)})
    keys = ["_rank", "distance", "a_session", "a_frame", "b_session", "b_frame"]
    ordered = pairs.assign(_rank=rank).sort_values(keys, kind="stable")
    return ordered.drop(columns="_rank").reset_index(drop=True)


def classify_pairs(
    frames: pd.DataFrame, found: NearPairs, training: Collection[str]
) -> pd.DataFrame:
    """The pairs that matter, one row each, with PAIR_COLUMNS, in report order.

    `frames` has FRAME_COLUMNS, one row per hash that `found` points into, and has passed
    `check_frames`. In the cross-split kinds a is the train/ frame and b the test/ frame; in
    training_vs_training a is the frame with the lower session id. Other pairs are dropped:
    two frames of one session (except a crossing session's train/ and test/ frames), a
    crossing session's train/ frame with a training session's frame, and two test/ frames.
    Report order is kind (as in KINDS), distance, then a's and b's session and frame.
    """
    split = frames["split"].astype(str).to_numpy()
    session = frames["session_id"].astype(str).to_numpy()
    number = frames["frame_number"].to_numpy(dtype=np.int64)
    codes, _ids = pd.factorize(frames["session_id"].astype(str), sort=True)
    code = np.asarray(codes)  # sessions as numbers in id order, for fast comparisons
    is_training = frames["session_id"].astype(str).isin(set(training)).to_numpy()

    i, j = found.first, found.second
    i_train, j_train = split[i] == "train", split[j] == "train"
    cross = i_train != j_train
    later = (code[i] > code[j]) | ((code[i] == code[j]) & (number[i] > number[j]))
    swap = np.where(cross, ~i_train, later)  # cross-split: the train/ frame goes first
    a, b = np.where(swap, j, i), np.where(swap, i, j)
    same = code[a] == code[b]
    a_training, b_training = is_training[a], is_training[b]

    kind = np.full(len(a), "", dtype=object)
    kind[cross & a_training] = TRAINING_VS_TEST
    kind[cross & ~a_training & same] = CROSSING_SAME_SESSION
    kind[cross & ~a_training & ~same] = CROSSING_VS_OTHER_TEST
    kind[i_train & j_train & a_training & b_training & ~same] = TRAINING_VS_TRAINING
    keep = kind != ""
    a, b = a[keep], b[keep]
    pairs = pd.DataFrame(
        {
            "kind": kind[keep].astype(str),
            "distance": found.distance[keep],
            "a_split": split[a],
            "a_session": session[a],
            "a_frame": number[a],
            "b_split": split[b],
            "b_session": session[b],
            "b_frame": number[b],
        },
        columns=list(PAIR_COLUMNS),
    )
    return _in_report_order(pairs)


def thumbnail_name(kind: str, a_session: str, a_frame: int, b_session: str, b_frame: int) -> str:
    return f"{kind}__{a_session}_f{a_frame:05d}__{b_session}_f{b_frame:05d}.png"


def with_thumbnails(pairs: pd.DataFrame) -> pd.DataFrame:
    """`pairs` with a thumbnail column: one file name per pair of sessions, else "".

    For each kind in CLASS_KINDS and each pair of sessions (a_session, b_session), its
    closest pair gets one: fewest bits, then the lowest a frame, then the lowest b frame.
    Names are relative to the folder of duplicates.csv.
    """
    out = pairs.copy()
    out["thumbnail"] = ""
    eligible = out[out["kind"].isin(CLASS_KINDS)]
    closest = eligible.sort_values(
        ["kind", "a_session", "b_session", "distance", "a_frame", "b_frame"], kind="stable"
    ).drop_duplicates(subset=["kind", "a_session", "b_session"], keep="first")
    names = [
        f"{THUMBNAIL_DIR}/{thumbnail_name(kind, a_s, int(a_f), b_s, int(b_f))}"
        for kind, a_s, a_f, b_s, b_f in zip(
            closest["kind"],
            closest["a_session"],
            closest["a_frame"],
            closest["b_session"],
            closest["b_frame"],
            strict=True,
        )
    ]
    out.loc[closest.index, "thumbnail"] = names
    return out


def thumbnail_frames(
    kind: str, a_session: str, a_frame: int, b_session: str, b_frame: int
) -> list[tuple[str, int]]:
    """The frames a pair's thumbnail shows, as (session id, frame number), left to right.

    For training_vs_test, the training frame alone: its test frame is never drawn (data
    rule 2). For training_vs_training, both frames. Raises ValueError for any other kind.
    """
    if kind == TRAINING_VS_TEST:
        return [(a_session, a_frame)]
    if kind == TRAINING_VS_TRAINING:
        return [(a_session, a_frame), (b_session, b_frame)]
    raise ValueError(f"{kind} pairs get no thumbnail")


def duplicates_csv(pairs: pd.DataFrame) -> str:
    """duplicates.csv: one row per pair, with CSV_COLUMNS. Labels never go in: some sessions
    are marked blind until onset pass 2."""
    return pairs.to_csv(index=False, columns=list(CSV_COLUMNS), lineterminator="\n")


def same_hash_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """The training_vs_test pairs whose hashes are identical: 0 bits apart.

    Byte-identical files have identical pixels and so identical hashes, so every exact copy
    of a training frame in test/ is among these pairs: only their files need a checksum.
    """
    return pairs[(pairs["kind"] == TRAINING_VS_TEST) & (pairs["distance"] == 0)]


@dataclass(frozen=True)
class ExactCopies:
    """The pairs of `same_hash_pairs`, compared byte for byte by SHA-256."""

    n_checked: int  # pairs compared
    n_train_files: int  # distinct train/ files checksummed
    n_test_files: int  # distinct test/ files checksummed
    identical: tuple[tuple[str, str], ...]  # (train/ path, test/ path) of byte-identical files


def exact_copies(paths: Sequence[tuple[str, str]], sha256: Mapping[str, str]) -> ExactCopies:
    """Which (train/ path, test/ path) pairs are byte-identical, given each file's SHA-256.

    Paths are relative to the release; the identical pairs come back sorted.
    """
    identical = sorted((train, test) for train, test in paths if sha256[train] == sha256[test])
    return ExactCopies(
        n_checked=len(paths),
        n_train_files=len({train for train, _test in paths}),
        n_test_files=len({test for _train, test in paths}),
        identical=tuple(identical),
    )


def _copies_section(copies: ExactCopies) -> list[str]:
    n, k = copies.n_checked, len(copies.identical)
    lines = [
        "## Exact copies",
        "",
        "Byte-identical files have identical pixels and so identical hashes, so every exact "
        f"copy of a training frame in test/ is among the {n:,} `{TRAINING_VS_TEST}` pairs 0 "
        f"bits apart. Their files, {copies.n_train_files:,} in train/ and "
        f"{copies.n_test_files:,} in test/, were compared by SHA-256, which data rule 2 allows "
        f"for the leakage audit: {k:,} of the {n:,} pairs are byte-identical.",
        "",
    ]
    if copies.identical:
        rows = [[f"`{train}`", f"`{test}`"] for train, test in copies.identical]
        lines += [markdown_table(["train/ file", "test/ file"], rows, [False, False]), ""]
    else:
        lines += ["No test file is a byte copy of a training file.", ""]
    lines.append(
        "SHA-256 finds byte copies only: a frame saved again with other PNG settings keeps its "
        "pixels but not its bytes, and would not count as a copy here."
    )
    return lines


@dataclass(frozen=True)
class ClassRow:
    """Flagged frames of one class, and all frames of that class that could be flagged."""

    label: int
    vs_test_train: int  # train/ frames in a training_vs_test pair
    vs_test_test: int  # test/ frames in a training_vs_test pair
    vs_training: int  # frames in a training_vs_training pair
    training_frames: int  # frames in the training sessions counted
    test_frames: int  # frames in test/


@dataclass(frozen=True)
class ClassTable:
    rows: tuple[ClassRow, ...]  # one per class 0-5
    counted: Mapping[str, int]  # pairs counted, per kind in CLASS_KINDS
    mixed: Mapping[str, int]  # counted pairs whose two frames have different classes
    n_left_out: int  # pairs of CLASS_KINDS with a frame in a marked session
    n_training_sessions: int  # training sessions counted: those not marked
    n_training_frames: int
    n_test_sessions: int  # sessions with frames in test/
    n_test_frames: int


def _labelled(pairs: pd.DataFrame, labels: pd.DataFrame, side: str) -> pd.DataFrame:
    """`pairs` with a `<side>_label` column. Raises ValueError if a frame has no label."""
    table = labels.rename(
        columns={
            "session_id": f"{side}_session",
            "frame_number": f"{side}_frame",
            "label": f"{side}_label",
        }
    )[[f"{side}_session", f"{side}_frame", f"{side}_label"]]
    merged = pairs.merge(table, on=[f"{side}_session", f"{side}_frame"], how="left")
    if merged[f"{side}_label"].isna().any():
        raise ValueError("a flagged frame has no row in the labels table")
    return merged


def _counts(labels: pd.Series) -> dict[int, int]:
    """Class -> number of rows."""
    counts = labels.astype(np.int64).value_counts()
    keys = counts.index.to_numpy(dtype=np.int64).tolist()
    return dict(zip(keys, counts.to_numpy().tolist(), strict=True))


def _per_class(frames: pd.DataFrame) -> dict[int, int]:
    """Distinct frames per class, from columns session, frame and label."""
    return _counts(frames.drop_duplicates(subset=["session", "frame"])["label"])


def _side(pairs: pd.DataFrame, side: str) -> pd.DataFrame:
    columns = [f"{side}_session", f"{side}_frame", f"{side}_label"]
    return pairs[columns].set_axis(["session", "frame", "label"], axis=1)


def class_table(
    pairs: pd.DataFrame,
    labels: pd.DataFrame,
    training: Collection[str],
    marked: Collection[str],
) -> ClassTable:
    """Flagged frames per class in the kinds of CLASS_KINDS, leaving out marked sessions.

    Every pair with a frame in a session of `marked` is left out and only counted.
    `labels` has LABEL_COLUMNS for the frames outside the marked sessions. A row of a marked
    session raises ValueError, so their labels can't be counted by mistake.
    """
    marked_set = set(marked)
    present = sorted(set(labels["session_id"].astype(str)) & marked_set)
    if present:
        raise ValueError(f"the labels table holds sessions marked blind: {', '.join(present)}")
    relevant = pairs[pairs["kind"].isin(CLASS_KINDS)]
    touches = relevant["a_session"].isin(marked_set) | relevant["b_session"].isin(marked_set)
    counted = _labelled(_labelled(relevant[~touches], labels, "a"), labels, "b")
    vs_test = counted[counted["kind"] == TRAINING_VS_TEST]
    vs_training = counted[counted["kind"] == TRAINING_VS_TRAINING]

    flagged_train = _per_class(_side(vs_test, "a"))
    flagged_test = _per_class(_side(vs_test, "b"))
    flagged_training = _per_class(pd.concat([_side(vs_training, "a"), _side(vs_training, "b")]))
    in_training = labels[labels["session_id"].astype(str).isin(set(training))]
    in_test = labels[labels["split"] == "test"]
    training_frames = _counts(in_training["label"])
    test_frames = _counts(in_test["label"])
    rows = tuple(
        ClassRow(
            label=label,
            vs_test_train=flagged_train.get(label, 0),
            vs_test_test=flagged_test.get(label, 0),
            vs_training=flagged_training.get(label, 0),
            training_frames=training_frames.get(label, 0),
            test_frames=test_frames.get(label, 0),
        )
        for label in CLASS_NAMES
    )
    return ClassTable(
        rows=rows,
        counted={TRAINING_VS_TEST: len(vs_test), TRAINING_VS_TRAINING: len(vs_training)},
        mixed={
            TRAINING_VS_TEST: int((vs_test["a_label"] != vs_test["b_label"]).sum()),
            TRAINING_VS_TRAINING: int((vs_training["a_label"] != vs_training["b_label"]).sum()),
        },
        n_left_out=int(touches.sum()),
        n_training_sessions=in_training["session_id"].nunique(),
        n_training_frames=len(in_training),
        n_test_sessions=in_test["session_id"].nunique(),
        n_test_frames=len(in_test),
    )


@dataclass(frozen=True)
class Scope:
    """What was hashed and searched, for the report."""

    n_frames: int
    n_sessions: int
    n_train_frames: int  # frames in train/
    n_test_frames: int  # frames in test/
    n_training_sessions: int
    n_test_sessions: int  # crossing ones included
    n_crossing_sessions: int
    n_marked: int  # sessions marked blind for onset pass 2
    fewest_bits: int  # fewest bits set in a hash
    most_bits: int
    max_distance: int
    n_candidates: int
    chunk_widths: tuple[int, ...]
    n_thumbnails: int


@dataclass(frozen=True)
class Provenance:
    """Where the report's numbers come from (data rule 8)."""

    code_git_sha: str
    index_git_sha: str
    dataset_fingerprint: str
    phash_algorithm: str


def _distances(scope: Scope) -> list[int]:
    """The distances a pair can have. Hashes with equal bit counts are an even distance apart."""
    step = 2 if scope.fewest_bits == scope.most_bits else 1
    return list(range(0, scope.max_distance + 1, step))


def _kind_table(pairs: pd.DataFrame, scope: Scope) -> str:
    distances = _distances(scope)
    rows: list[list[str]] = []
    for kind in KINDS:
        mine = pairs[pairs["kind"] == kind]
        per_distance = mine["distance"].value_counts()
        rows.append(
            [
                f"`{kind}`",
                f"{len(mine):,}",
                *(f"{int(per_distance.get(d, 0)):,}" for d in distances),
                f"{len(mine.drop_duplicates(subset=['a_session', 'a_frame'])):,}",
                f"{mine['a_session'].nunique():,}",
                f"{len(mine.drop_duplicates(subset=['b_session', 'b_frame'])):,}",
                f"{mine['b_session'].nunique():,}",
            ]
        )
    header = [
        "Kind",
        "Pairs",
        *(f"d = {d}" for d in distances),
        "Frames a",
        "Sessions a",
        "Frames b",
        "Sessions b",
    ]
    return markdown_table(header, rows, [False] + [True] * (len(header) - 1))


def _session_pair_table(pairs: pd.DataFrame) -> str | None:
    relevant = pairs[pairs["kind"].isin(CLASS_KINDS)]
    if relevant.empty:
        return None
    grouped = (
        relevant.groupby(["kind", "a_session", "b_session"], sort=True)["distance"]
        .agg(["size", "min"])
        .reset_index()
    )
    rank = grouped["kind"].map({kind: n for n, kind in enumerate(KINDS)})
    grouped = grouped.assign(_rank=rank, _neg=-grouped["size"]).sort_values(
        ["_rank", "_neg", "a_session", "b_session"], kind="stable"
    )
    top = grouped.groupby("kind", sort=False).head(TOP_SESSION_PAIRS)
    rows = [
        [f"`{kind}`", a, b, f"{int(n):,}", str(int(fewest))]
        for kind, a, b, n, fewest in zip(
            top["kind"], top["a_session"], top["b_session"], top["size"], top["min"], strict=True
        )
    ]
    header = ["Kind", "Session a", "Session b", "Pairs", "Fewest bits"]
    return markdown_table(header, rows, [False, False, False, True, True])


def _class_section(classes: ClassTable, scope: Scope) -> list[str]:
    rows = [
        [
            f"{row.label} {CLASS_NAMES[row.label]}",
            f"{row.vs_test_train:,} of {row.training_frames:,}",
            f"{row.vs_test_test:,} of {row.test_frames:,}",
            f"{row.vs_training:,} of {row.training_frames:,}",
        ]
        for row in classes.rows
    ]
    header = [
        "Class",
        "`training_vs_test`: train/ frames",
        "`training_vs_test`: test/ frames",
        "`training_vs_training`: frames",
    ]
    total = classes.n_left_out + sum(classes.counted.values())
    vs_test, vs_training = TRAINING_VS_TEST, TRAINING_VS_TRAINING
    return [
        "## Flagged frames by class",
        "",
        f"Left out until onset pass 2 is done: {classes.n_left_out:,} of the {total:,} "
        f"`{vs_test}` and `{vs_training}` pairs, those with a frame in a session marked "
        f"blind ({scope.n_marked} marked). Their labels were not read.",
        "",
        markdown_table(header, rows, [False, True, True, True]),
        "",
        "- A cell counts the frames of that class in at least one counted pair, out of all "
        f"frames of that class: in the {classes.n_training_sessions} training sessions not "
        f"marked blind ({classes.n_training_frames:,} frames), or in test/ "
        f"({classes.n_test_frames:,} frames, {classes.n_test_sessions} sessions). A frame has "
        "one class while a pair can have two, so frames are counted, not pairs.",
        "- Counted pairs whose two frames have different classes: "
        f"{classes.mixed[vs_test]:,} of {classes.counted[vs_test]:,} `{vs_test}` and "
        f"{classes.mixed[vs_training]:,} of {classes.counted[vs_training]:,} `{vs_training}`. "
        "Near-identical frames with different labels are leads for the label audit.",
    ]


def duplicates_markdown(
    pairs: pd.DataFrame,
    classes: ClassTable,
    copies: ExactCopies,
    scope: Scope,
    provenance: Provenance,
) -> str:
    """duplicates.md: pairs by kind, exact copies, the busiest session pairs, flagged frames
    by class, the search and its cost, and where the numbers come from."""
    if scope.fewest_bits == scope.most_bits:
        bits = f"Every hash has {scope.fewest_bits} bits set, so every distance is even."
    else:
        bits = f"Hashes have from {scope.fewest_bits} to {scope.most_bits} bits set."
    widths = ", ".join(str(w) for w in scope.chunk_widths)
    all_pairs = scope.n_frames * (scope.n_frames - 1) // 2
    share = 100 * scope.n_candidates / all_pairs if all_pairs else 0.0
    session_pairs = _session_pair_table(pairs)
    lines = [
        "# Near-duplicate audit: perceptual hashes",
        "",
        f"Every frame of the release was hashed: {scope.n_frames:,} frames from "
        f"{scope.n_sessions} sessions, {scope.n_train_frames:,} in train/ and "
        f"{scope.n_test_frames:,} in test/. Two frames are flagged when their 64-bit DCT "
        f"perceptual hashes differ in at most {scope.max_distance} bits. {bits} The "
        f"{scope.n_training_sessions} training sessions and {scope.n_test_sessions} test "
        f"sessions ({scope.n_crossing_sessions} of them crossing) come from "
        "`weldpilot.data.splits` (data rule 4). Test frames were hashed, and those in pairs "
        "with identical hashes checksummed, but never shown (data rule 2). Counts are exact "
        "for this release (a census, not an estimate), so they carry no interval.",
        "",
        "## Pairs by kind",
        "",
        _kind_table(pairs, scope),
        "",
        f"- `{TRAINING_VS_TEST}`: a is a frame of a training session, b a test/ frame. This is "
        "the leakage that matters for the final test.",
        f"- `{CROSSING_SAME_SESSION}`: a is a crossing session's train/ frame, b a test/ frame "
        "of the same session: one recording, which is why data rule 3 keeps crossing "
        "sessions out of training.",
        f"- `{CROSSING_VS_OTHER_TEST}`: a is a crossing session's train/ frame, b a test/ frame "
        "of another session. Crossing frames never train, so this is not leakage.",
        f"- `{TRAINING_VS_TRAINING}`: a and b are frames of two different training sessions. "
        "Such pairs make session-level folds less independent.",
        "- Not reported: two frames of one session (other than the above), a crossing "
        "session's train/ frame with a training session's frame, and two test/ frames.",
        "",
        *_copies_section(copies),
        "",
        "## Session pairs with the most pairs",
        "",
        session_pairs or "No pairs of these kinds.",
        "",
        f"Up to {TOP_SESSION_PAIRS} per kind; duplicates.csv lists every pair.",
        "",
        *_class_section(classes, scope),
        "",
        "## Search and cost",
        "",
        f"Comparing every pair of the {scope.n_frames:,} hashes would take {all_pairs:,} "
        f"checks. Instead each hash was cut into {len(scope.chunk_widths)} chunks of {widths} "
        f"bits. Two hashes at most {scope.max_distance} bits apart differ in at most "
        f"{scope.max_distance} chunks, so they agree exactly on at least one: only hashes that "
        f"share a chunk's value were compared, on all {HASH_BITS} bits. That took "
        f"{scope.n_candidates:,} checks ({share:.2f}% of all pairs) and finds every pair within "
        f"{scope.max_distance} bits.",
        "",
        "## Thumbnails",
        "",
        f"For each pair of sessions of kind `{TRAINING_VS_TEST}` or `{TRAINING_VS_TRAINING}`, "
        f"its closest pair (fewest bits, then lowest frame numbers) has a thumbnail in "
        f"{THUMBNAIL_DIR}/: the training frame alone for `{TRAINING_VS_TEST}`, and both "
        f"frames side by side, a on the left, for `{TRAINING_VS_TRAINING}`. Test frames are "
        f"never drawn (data rule 2). {scope.n_thumbnails:,} files; duplicates.csv names each "
        "row's thumbnail.",
        "",
        "## Provenance",
        "",
        f"- Code: git `{provenance.code_git_sha}`.",
        f"- Index: built at git `{provenance.index_git_sha}` from the dataset with fingerprint "
        f"`{provenance.dataset_fingerprint}`.",
        f"- Hash: {provenance.phash_algorithm}.",
        f"- Parameters: pairs at most {scope.max_distance} bits apart; "
        f"{len(scope.chunk_widths)} chunks of {widths} bits; SHA-256 of the files of "
        f"`{TRAINING_VS_TEST}` pairs 0 bits apart; one thumbnail per pair of sessions. No "
        "randomness, so no seed. The command has no config file, so these are all its "
        "parameters.",
        "- Command: `weldpilot audit duplicates`.",
    ]
    return "\n".join(lines) + "\n"
