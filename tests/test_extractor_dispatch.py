"""Tests for Stage-1 (column extraction) dispatch via ``ExtractorConfig``.

These tests verify the registry's ``build_extractor_command`` factory produces
the right subprocess command for each model in ``_MODEL_EXTRACTORS``. They are
pure command-construction checks — no GPU, no model load — so they live in the
default (not-slow) test suite.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

from trl_bench.registry import (
    build_extractor_command,
    build_query_extractor_command,
    build_table_encoder_command,
    ExtractorConfig,
    QueryEncoderConfig,
    TableEncoderConfig,
    SettingError,
    _MODEL_EXTRACTORS,
    _QUERY_ENCODER_EXTRACTORS,
    _TABLE_ENCODERS,
)


# == build_extractor_command: per-model command shape ========================

def test_bert_extractor_command_has_input_output_and_device(tmp_path):
    """BERT runs the in-repo bert wrapper with --input/--output and --device cuda."""
    csv_dir = tmp_path / "tables_all"
    out_pkl = tmp_path / "bert_spider_join.pkl"
    cmd = build_extractor_command(
        model="bert", dataset="spider_join",
        input_dir=csv_dir, output_path=out_pkl,
    )

    # Canonical invocation: python -m trl_bench.models.bert.generate_column_embeddings ...
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.models.bert.generate_column_embeddings"]
    args = dict(zip(cmd[3::2], cmd[4::2]))
    assert args["--input"] == str(csv_dir)
    assert args["--output"] == str(out_pkl)
    assert args["--device"] == "cuda"
    # BERT has no extra args today — the wrapper's --model default (bert-base-uncased)
    # is the paper checkpoint, so we don't override it.
    assert "--model" not in args


def test_gte_extractor_command_pins_gte_base_checkpoint(tmp_path):
    """GTE invokes the gte wrapper and pins --model thenlper/gte-base."""
    csv_dir = tmp_path / "tables_all"
    out_pkl = tmp_path / "gte_spider_join.pkl"
    cmd = build_extractor_command(
        model="gte", dataset="spider_join",
        input_dir=csv_dir, output_path=out_pkl,
    )

    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.models.gte.generate_column_embeddings"]
    args = dict(zip(cmd[3::2], cmd[4::2]))
    assert args["--input"] == str(csv_dir)
    assert args["--output"] == str(out_pkl)
    assert args["--model"] == "thenlper/gte-base"   # paper checkpoint pin
    assert args["--device"] == "cuda"


def test_tapas_extractor_command_uses_argparse_default_checkpoint(tmp_path):
    """TAPAS runs the in-repo tapas wrapper with --input/--output, --device cuda.

    The wrapper's argparse default for ``--model`` is ``google/tapas-base`` —
    the paper checkpoint — so the registry does NOT pass an explicit --model
    override (mirrors the BERT wiring decision).
    """
    csv_dir = tmp_path / "tables_all"
    out_pkl = tmp_path / "tapas_spider_join.pkl"
    cmd = build_extractor_command(
        model="tapas", dataset="spider_join",
        input_dir=csv_dir, output_path=out_pkl,
    )

    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.models.tapas.generate_column_embeddings"]
    args = dict(zip(cmd[3::2], cmd[4::2]))
    assert args["--input"] == str(csv_dir)
    assert args["--output"] == str(out_pkl)
    assert args["--device"] == "cuda"
    # No --model override — TAPAS wrapper's argparse default is the paper
    # checkpoint google/tapas-base.
    assert "--model" not in args


def test_extractor_command_overrides_device(tmp_path):
    """Passing device='cpu' overrides ExtractorConfig.device_value."""
    cmd = build_extractor_command(
        model="bert", dataset="spider_join",
        input_dir=tmp_path / "tables_all",
        output_path=tmp_path / "out.pkl",
        device="cpu",
    )
    args = dict(zip(cmd[3::2], cmd[4::2]))
    assert args["--device"] == "cpu"


def test_unknown_model_raises_pointing_at_registry_constant(tmp_path):
    """Unwired models raise a clear error naming `_MODEL_EXTRACTORS`.

    Uses a clearly bogus model name — the historical examples (TAPAS, then
    TaBERT) have since been wired into ``_MODEL_EXTRACTORS``, so we now use
    a sentinel string that the registry will never contain.
    """
    with pytest.raises(SettingError, match="_MODEL_EXTRACTORS"):
        build_extractor_command(
            model="nonexistent_model_xyz", dataset="spider_join",
            input_dir=tmp_path / "tables_all",
            output_path=tmp_path / "out.pkl",
        )


# == _MODEL_EXTRACTORS: invariants ==========================================

def test_model_extractors_runner_paths_are_importable_module_names():
    """Each runner must be a dotted Python module path (no .py suffix)."""
    for model, cfg in _MODEL_EXTRACTORS.items():
        assert "/" not in cfg.runner, f"{model}: runner must be dotted, not a path"
        assert not cfg.runner.endswith(".py"), f"{model}: drop .py suffix"
        # The runner module is what `python -m <runner>` will load.
        assert cfg.runner.startswith("trl_bench."), (
            f"{model}: runner must live under trl_bench.* for `python -m` to "
            f"resolve it after `pip install -e .`"
        )


def test_extractor_commands_emit_only_flags_their_runner_accepts(monkeypatch):
    """``build_extractor_command`` must emit only --flags the TARGET runner's
    argparse declares -- for table extractors AND record_linkage row overrides,
    across EVERY model.

    Covers ALL FIVE Stage-1 dispatch surfaces: table extractors, record_linkage
    row overrides, table-direct encoders, and query encoders.

    Regression (real bugs, commits f15a8e5/bfef1eb + follow-up): the dispatchers
    emit per-config flags that a swapped/outlier runner does not accept.
    bert/gte/openai/tabbie table extractors use ``--input``/``--output`` but
    their row runners use ``--input_dir``/``--output_path`` -> record_linkage
    Stage-1 argparse-failed; tuta's row entry pointed at split-aware
    row_prediction tooling (``--dataset_dir``); openai's query encoder got
    ``--mode`` which the OpenAI-API runner has no concept of. All are caught by
    checking emitted-flags ⊆ declared-flags for the TARGET runner -- the exact
    invariant the original (source-substring) test failed to encode.
    """
    import re
    import trl_bench
    from trl_bench.registry import (
        build_extractor_command, build_table_encoder_command,
        build_query_extractor_command, build_question_extractor_command,
        _MODEL_EXTRACTORS, _ROW_RUNNERS,
        _TABLE_ENCODERS, _QUERY_ENCODER_EXTRACTORS,
    )
    # Checkpoint-gated wrappers (tuta/tabbie/...) raise SettingError on hosts
    # without the licensed .bin; we only inspect FLAGS, so bypass the existence
    # check -- that exact gap let the record_linkage bugs ship untested.
    monkeypatch.setattr(Path, "exists", lambda self: True)
    pkg_root = Path(trl_bench.__file__).resolve().parent

    def declared(runner):
        rel = runner.split(".", 1)[1].replace(".", "/") + ".py"
        return set(re.findall(r'add_argument\(\s*["\'](--[\w\-]+)',
                              (pkg_root / rel).read_text()))

    def emitted(cmd):
        return [a for a in cmd[3:] if a.startswith("--")]

    common = dict(dataset="ds", input_dir=Path("/in"),
                  output_path=Path("/out.pkl"), device="cuda:0",
                  checkpoint_root="/ck")

    for model, cfg in _MODEL_EXTRACTORS.items():
        cmd = build_extractor_command(model=model, **common)
        miss = [f for f in emitted(cmd) if f not in declared(cfg.runner)]
        assert not miss, f"[extractor] {model}: emits {miss} not in {cfg.runner}"

    for model, runner in _ROW_RUNNERS.items():
        cmd = build_extractor_command(model=model, runner_override=runner, **common)
        miss = [f for f in emitted(cmd) if f not in declared(runner)]
        assert not miss, (f"[row] {model}: record_linkage Stage-1 emits {miss} "
                          f"not declared by {runner}")

    for model, cfg in _TABLE_ENCODERS.items():
        cmd = build_table_encoder_command(model=model, dataset="ds",
                                          input_dir=Path("/in"), output_path=Path("/o.pkl"))
        miss = [f for f in emitted(cmd) if f not in declared(cfg.runner)]
        assert not miss, f"[tabenc] {model}: emits {miss} not declared by {cfg.runner}"

    for model, cfg in _QUERY_ENCODER_EXTRACTORS.items():
        cmd = build_query_extractor_command(model=model, input_json=Path("/in.json"),
                                            output_path=Path("/o.pkl"))
        miss = [f for f in emitted(cmd) if f not in declared(cfg.runner)]
        assert not miss, f"[qryenc] {model}: emits {miss} not declared by {cfg.runner}"

    # semantic_parsing QUESTION encoder (token mode); only sentence_t5/mpnet.
    from trl_bench.registry import _SEMPARSE_QUESTION_ENCODERS
    for model in sorted(_SEMPARSE_QUESTION_ENCODERS):
        cfg = _QUERY_ENCODER_EXTRACTORS[model]
        cmd = build_question_extractor_command(model=model, input_json=Path("/q.jsonl"),
                                               output_path=Path("/o.pkl"))
        miss = [f for f in emitted(cmd) if f not in declared(cfg.runner)]
        assert not miss, f"[qstenc] {model}: emits {miss} not declared by {cfg.runner}"


def test_build_question_extractor_command_uses_token_mode_for_semparse():
    """semantic_parsing question embeddings come ONLY from the benchmark's
    designated encoders {sentence_t5, mpnet} (the model under test supplies
    COLUMN embeddings, not questions). Those emit per-token ``--mode token``,
    ``--tokens_field tokens``, ``--id_field id``. Any other ``--setting`` --
    token-capable models (bert/gte), openai (cls-only), or table models --
    raises ``SettingError`` directing to sentence_t5/mpnet. Bug #6 regression."""
    from trl_bench.registry import build_question_extractor_command
    for model in ("sentence_t5", "mpnet"):
        cmd = build_question_extractor_command(
            model=model, input_json=Path("/q/train_split.jsonl"),
            output_path=Path("/o/questions_train.pkl"))
        assert cmd[cmd.index("--mode") + 1] == "token"
        assert cmd[cmd.index("--tokens_field") + 1] == "tokens"
        assert cmd[cmd.index("--id_field") + 1] == "id"
        assert cmd[cmd.index("--output") + 1] == "/o/questions_train.pkl"
    for bad in ("bert", "gte", "openai", "tapas"):
        with pytest.raises(SettingError, match="sentence_t5"):
            build_question_extractor_command(
                model=bad, input_json=Path("/q/train_split.jsonl"),
                output_path=Path("/o/questions_train.pkl"))


def test_tuta_csv_to_embeddings_importers_put_package_dir_on_path():
    """Any tuta runner that bare-imports ``csv_to_embeddings`` must add THIS
    package dir (models/tuta/) to ``sys.path`` first, else ``python -m <runner>``
    raises ``ModuleNotFoundError: No module named 'csv_to_embeddings'``.

    Regression: generate_row_embeddings.py inserted ``../../`` (= models/) rather
    than ``os.path.dirname(__file__)`` (= models/tuta/), breaking the bare import
    under ``pip install -e .``.
    """
    import trl_bench

    tuta_dir = Path(trl_bench.__file__).resolve().parent / "models" / "tuta"
    importers = [
        py for py in sorted(tuta_dir.glob("generate_*.py"))
        if "from csv_to_embeddings import" in py.read_text()
    ]
    assert importers, "expected >=1 tuta runner importing csv_to_embeddings"
    for py in importers:
        src = py.read_text()
        assert "sys.path.insert(0, os.path.dirname(__file__))" in src, (
            f"{py.name} bare-imports csv_to_embeddings but does not put its own "
            f"package dir on sys.path; `python -m {py.stem}` will fail to resolve "
            f"it after `pip install -e .`."
        )


def test_extractor_config_extra_args_is_tuple_of_pairs():
    """extra_args is tuple-of-tuples (immutable, frozen-dataclass-safe)."""
    for model, cfg in _MODEL_EXTRACTORS.items():
        assert isinstance(cfg.extra_args, tuple), f"{model}: extra_args must be tuple"
        for entry in cfg.extra_args:
            assert isinstance(entry, tuple) and len(entry) == 2, (
                f"{model}: each extra_args entry must be (flag, value)"
            )


def test_extractor_config_derived_args_is_tuple_of_pairs():
    """derived_args is tuple-of-tuples (templated values resolved at build time)."""
    for model, cfg in _MODEL_EXTRACTORS.items():
        assert isinstance(cfg.derived_args, tuple), (
            f"{model}: derived_args must be tuple"
        )
        for entry in cfg.derived_args:
            assert isinstance(entry, tuple) and len(entry) == 2, (
                f"{model}: each derived_args entry must be (flag, template)"
            )


def test_all_wired_extractors_match_model_granularities():
    """Every model in ``_MODEL_EXTRACTORS`` must appear in ``_MODEL_GRANULARITIES``.

    If a Stage-1 extractor is wired for a model not in the capability table, the
    caller will get a confusing "model has no granularity" error elsewhere.
    """
    from trl_bench.registry import _MODEL_GRANULARITIES
    for model in _MODEL_EXTRACTORS:
        assert model in _MODEL_GRANULARITIES, (
            f"{model!r} wired in _MODEL_EXTRACTORS but missing from "
            f"_MODEL_GRANULARITIES"
        )


# == Per-wrapper command-shape tests ========================================
# Each test pins the exact CLI surface the registry dispatches for one model,
# matching the wrapper's argparse signature (or USAGE.md). The "(model dataset
# extractor) -> command" mapping is the entire contract surface that auto-
# orchestration depends on; a regression here would silently produce a wrong
# subprocess command on the user's host.

def _args_dict_from_cmd(cmd):
    """Parse a ``python -m <runner> --flag value [--flag value ...]`` cmd.

    Returns a {flag -> value} dict. Assumes alternating flag/value pairs,
    which is the contract of ``build_extractor_command``.
    """
    assert cmd[:2] == [sys.executable, "-m"]
    flat = cmd[3:]
    return dict(zip(flat[0::2], flat[1::2]))


def test_openai_extractor_command_uses_bert_shape_no_device(tmp_path):
    """OpenAI: --input/--output (BERT shape) with NO --device flag.

    The wrapper routes through the OpenAI HTTP API (no GPU); we deliberately
    omit ``--device`` to match its argparse surface. The argparse default for
    ``--model`` (``text-embedding-3-small``) is the paper checkpoint, so no
    --model override is needed (mirrors BERT/TAPAS).
    """
    csv_dir = tmp_path / "tables_all"
    out_pkl = tmp_path / "openai_spider_join.pkl"
    cmd = build_extractor_command(
        model="openai", dataset="spider_join",
        input_dir=csv_dir, output_path=out_pkl,
    )
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.models.openai.generate_column_embeddings"]
    args = _args_dict_from_cmd(cmd)
    assert args["--input"] == str(csv_dir)
    assert args["--output"] == str(out_pkl)
    assert "--device" not in args     # API-only wrapper has no --device flag
    assert "--model" not in args      # use the wrapper's paper-default model id


def test_openai_extractor_command_ignores_device_override(tmp_path):
    """openai has device_flag=None -> passing device='cpu' is a silent no-op."""
    cmd = build_extractor_command(
        model="openai", dataset="spider_join",
        input_dir=tmp_path / "tables_all",
        output_path=tmp_path / "out.pkl",
        device="cpu",
    )
    assert "--device" not in cmd      # the flag never appears


def test_tabicl_extractor_command_uses_dir_shape_with_cuda(tmp_path):
    """TabICL: --input_dir/--output_path + --device cuda (overrides 'auto')."""
    csv_dir = tmp_path / "tables_all"
    out_pkl = tmp_path / "tabicl_openml.pkl"
    cmd = build_extractor_command(
        model="tabicl", dataset="openml_3",
        input_dir=csv_dir, output_path=out_pkl,
    )
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.models.tabicl.generate_embeddings_directory"]
    args = _args_dict_from_cmd(cmd)
    assert args["--input_dir"] == str(csv_dir)
    assert args["--output_path"] == str(out_pkl)
    assert args["--device"] == "cuda"
    assert "--input" not in args      # wrapper uses --input_dir, not --input
    assert "--output" not in args


def test_tabpfn_extractor_command_uses_dir_shape_with_cuda(tmp_path):
    """TabPFN: --input_dir/--output_path + --device cuda."""
    csv_dir = tmp_path / "tables_all"
    out_pkl = tmp_path / "tabpfn_openml.pkl"
    cmd = build_extractor_command(
        model="tabpfn", dataset="openml_3",
        input_dir=csv_dir, output_path=out_pkl,
    )
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.models.tabpfn.generate_embeddings_directory"]
    args = _args_dict_from_cmd(cmd)
    assert args["--input_dir"] == str(csv_dir)
    assert args["--output_path"] == str(out_pkl)
    assert args["--device"] == "cuda"


@pytest.mark.parametrize("model", [
    "transtab", "dae", "scarf", "vime", "subtab", "saint",
    "tabular_binning", "tabtransformer",
])
def test_trained_row_extractor_emits_checkpoint_base_dir(model, tmp_path):
    """Trained-per-table row models: --input_dir/--output_path + --checkpoint_base_dir.

    The wrappers train a fresh self-supervised model per CSV and write per-
    table checkpoints under ``--checkpoint_base_dir/<table_id>/`` (each
    cleaned up via ``shutil.rmtree`` after that table is embedded
    successfully, unless --keep_checkpoints). The dispatcher derives the
    base dir as ``{output_parent}/_ckpts_{output_stem}`` so it is a
    dedicated subdir scoped to this (model, dataset) pair — the rmtree-per-
    table cannot touch the output pickle or any sibling artefacts.

    These wrappers have NO --device flag (no GPU/CPU CLI option; device
    selection happens internally via the trainer config).
    """
    csv_dir = tmp_path / "tables_all"
    out_pkl = tmp_path / "per_cell" / f"{model}_openml.pkl"
    out_pkl.parent.mkdir(parents=True)
    cmd = build_extractor_command(
        model=model, dataset="openml_3",
        input_dir=csv_dir, output_path=out_pkl,
    )
    expected_runner = f"trl_bench.models.{model}.generate_embeddings_directory"
    assert cmd[:3] == [sys.executable, "-m", expected_runner]
    args = _args_dict_from_cmd(cmd)
    assert args["--input_dir"] == str(csv_dir)
    assert args["--output_path"] == str(out_pkl)
    # Derived per-cell scratch: dedicated subdir under the output's parent.
    # Stem ``<model>_openml`` keeps the dir name informative for debugging.
    expected_ckpt_dir = f"{out_pkl.parent}/_ckpts_{out_pkl.stem}"
    assert args["--checkpoint_base_dir"] == expected_ckpt_dir
    # The scratch dir MUST NOT be the same as the output pickle's parent —
    # that would put the cleanup rmtree(<base>/<table_id>) calls in the same
    # directory as the output pickle and any sibling datasets' artefacts.
    assert args["--checkpoint_base_dir"] != str(out_pkl.parent)
    assert "--device" not in args         # no --device flag on these wrappers


def test_trained_row_extractor_derived_checkpoint_dir_per_cell(tmp_path):
    """Different output paths -> different --checkpoint_base_dir values.

    The derived-value template is resolved at build_extractor_command time
    against both ``{output_parent}`` and ``{output_stem}``, so two
    (model, dataset) cells writing under the same parent dir get DIFFERENT
    checkpoint scratch dirs — preventing parallel runs from colliding on
    each other's training state.
    """
    shared_parent = tmp_path / "column" / "scarf"
    shared_parent.mkdir(parents=True)
    cell_a = shared_parent / "ds_a.pkl"   # same parent, different stem
    cell_b = shared_parent / "ds_b.pkl"

    cmd_a = build_extractor_command(
        model="scarf", dataset="ds_a",
        input_dir=tmp_path / "csvs", output_path=cell_a,
    )
    cmd_b = build_extractor_command(
        model="scarf", dataset="ds_b",
        input_dir=tmp_path / "csvs", output_path=cell_b,
    )
    args_a = _args_dict_from_cmd(cmd_a)
    args_b = _args_dict_from_cmd(cmd_b)
    assert args_a["--checkpoint_base_dir"] == f"{shared_parent}/_ckpts_ds_a"
    assert args_b["--checkpoint_base_dir"] == f"{shared_parent}/_ckpts_ds_b"
    assert args_a["--checkpoint_base_dir"] != args_b["--checkpoint_base_dir"]


# == Per-wrapper command-shape tests: licensed-checkpoint wrappers ==========
# These wrappers consume a host-specific checkpoint file/dir resolved against
# ``checkpoint_root`` (default ``./checkpoints``). The dispatcher appends
# ``<cfg.checkpoint_arg> <checkpoint_root>/<resolved-template>`` and raises
# ``SettingError`` (when ``checkpoint_required=True``) if the file is missing.
# The tests build a fake checkpoint under ``tmp_path/ckpts`` so the missing-
# file guard doesn't trip; a dedicated missing-checkpoint test pins the
# failure message shape.

def _touch_ckpt(path):
    """Create an empty placeholder file at ``path`` (parent dirs included)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_tabert_extractor_command_includes_resolved_checkpoint_path(tmp_path):
    """TaBERT: BERT-shape (--input / --output / --device) + --checkpoint
    <ckpt-root>/tabert/tabert_base_k3/model.bin.
    """
    ckpt_root = tmp_path / "ckpts"
    _touch_ckpt(ckpt_root / "tabert" / "tabert_base_k3" / "model.bin")
    csv_dir = tmp_path / "tables_all"
    out_pkl = tmp_path / "tabert_spider_join.pkl"
    cmd = build_extractor_command(
        model="tabert", dataset="spider_join",
        input_dir=csv_dir, output_path=out_pkl,
        checkpoint_root=ckpt_root,
    )
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.models.tabert.generate_column_embeddings"]
    args = _args_dict_from_cmd(cmd)
    assert args["--input"] == str(csv_dir)
    assert args["--output"] == str(out_pkl)
    assert args["--checkpoint"] == str(
        ckpt_root / "tabert" / "tabert_base_k3" / "model.bin"
    )
    assert args["--device"] == "cuda"


def test_tabsketchfm_extractor_command_includes_resolved_checkpoint(tmp_path):
    """TabSketchFM: BERT-shape + --checkpoint <ckpt-root>/tabsketchfm/<.ckpt>."""
    ckpt_root = tmp_path / "ckpts"
    _touch_ckpt(ckpt_root / "tabsketchfm" / "epoch=10-step=27786.ckpt")
    csv_dir = tmp_path / "tables_all"
    out_pkl = tmp_path / "tabsketchfm_spider_join.pkl"
    cmd = build_extractor_command(
        model="tabsketchfm", dataset="spider_join",
        input_dir=csv_dir, output_path=out_pkl,
        checkpoint_root=ckpt_root,
    )
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.models.tabsketchfm.generate_column_embeddings"]
    args = _args_dict_from_cmd(cmd)
    assert args["--input"] == str(csv_dir)
    assert args["--output"] == str(out_pkl)
    assert args["--checkpoint"] == str(
        ckpt_root / "tabsketchfm" / "epoch=10-step=27786.ckpt"
    )
    assert args["--device"] == "cuda"


def test_turl_extractor_command_uses_output_file_and_table_directory_mode(tmp_path):
    """TURL: --input_dir / --output_file (NOTE: --output_file, not _path)
    + --mode table_directory + --checkpoint <ckpt-root>/turl/pretrained.
    """
    ckpt_root = tmp_path / "ckpts"
    (ckpt_root / "turl" / "pretrained").mkdir(parents=True)
    csv_dir = tmp_path / "tables_all"
    out_pkl = tmp_path / "turl_valentine.pkl"
    cmd = build_extractor_command(
        model="turl", dataset="valentine",
        input_dir=csv_dir, output_path=out_pkl,
        checkpoint_root=ckpt_root,
    )
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.models.turl.generate_column_embeddings_dataset"]
    args = _args_dict_from_cmd(cmd)
    assert args["--input_dir"] == str(csv_dir)
    assert args["--output_file"] == str(out_pkl)
    assert "--output_path" not in args     # TURL uses --output_file
    assert args["--mode"] == "table_directory"
    assert args["--checkpoint"] == str(ckpt_root / "turl" / "pretrained")
    assert args["--device"] == "cuda"


def test_tuta_extractor_command_uses_device_id_and_model_path(tmp_path):
    """TUTA: --input_dir / --output_path + --model_path <.bin> + --device_id 0.

    The ``--device_id`` flag takes an int (NOT cuda/cpu); the dispatcher's
    ``device_value_map`` translates cuda->"0" by default. A subsequent test
    pins the cpu->"-1" translation.
    """
    ckpt_root = tmp_path / "ckpts"
    _touch_ckpt(ckpt_root / "tuta" / "tuta.bin")
    csv_dir = tmp_path / "tables_all"
    out_pkl = tmp_path / "tuta_opendata.pkl"
    cmd = build_extractor_command(
        model="tuta", dataset="opendata",
        input_dir=csv_dir, output_path=out_pkl,
        checkpoint_root=ckpt_root,
    )
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.models.tuta.generate_embeddings_directory"]
    args = _args_dict_from_cmd(cmd)
    assert args["--input_dir"] == str(csv_dir)
    assert args["--output_path"] == str(out_pkl)
    assert args["--model_path"] == str(ckpt_root / "tuta" / "tuta.bin")
    assert args["--device_id"] == "0"     # cuda -> "0" via device_value_map
    assert "--device" not in args         # wrapper has no --device flag
    assert "--checkpoint" not in args     # checkpoint is on --model_path


def test_tuta_extractor_command_translates_cpu_to_minus_one(tmp_path):
    """TUTA: device='cpu' override -> --device_id -1 via device_value_map."""
    ckpt_root = tmp_path / "ckpts"
    _touch_ckpt(ckpt_root / "tuta" / "tuta.bin")
    cmd = build_extractor_command(
        model="tuta", dataset="opendata",
        input_dir=tmp_path / "tables_all", output_path=tmp_path / "out.pkl",
        device="cpu", checkpoint_root=ckpt_root,
    )
    args = _args_dict_from_cmd(cmd)
    assert args["--device_id"] == "-1"


# == TABLE-native runner override (tuta) =====================================
# Regression: tuta's only Stage-1 runner (generate_embeddings_directory) emits
# per-ROW [CLS] embeddings (key ``row_embeddings``). The Stage-2 aggregator
# (generate_table_embeddings) reads ``column_embeddings``/``cls_embedding`` from
# the column pickle to build the table pickle's ``table_embedding`` dict -- tuta
# has neither, so cls_embedding/column_mean came out None and EVERY table-level
# cls cell (join/union classification, union_regression, table_subset,
# table_retrieval) failed with "embedding_type='cls' ... 'cls_embedding' is
# None". The fix: route tuta's TABLE-level extraction through a native runner
# (generate_table_embeddings_native) that writes the table pickle DIRECTLY with
# a populated ``cls_embedding`` (TUTA's native [CLS] token, the representation
# the paper used -- canonical_ref shows embedding_type="cls" for all 5 cells).

def test_table_native_runners_wires_tuta_to_native_table_extractor():
    """``_TABLE_NATIVE_RUNNERS`` maps tuta to its native table extractor.

    Models in this dict produce a table-level pickle (with a populated
    ``cls_embedding``) in a single native forward pass, bypassing the
    column-extraction + Stage-2-aggregator path that cannot derive a CLS
    embedding from tuta's row-only column pickle.
    """
    from trl_bench.registry import _TABLE_NATIVE_RUNNERS
    assert _TABLE_NATIVE_RUNNERS.get("tuta") == (
        "trl_bench.models.tuta.generate_table_embeddings_native"
    )
    # Runner must be a dotted, importable module path (python -m ...).
    runner = _TABLE_NATIVE_RUNNERS["tuta"]
    assert "/" not in runner and not runner.endswith(".py")
    assert runner.startswith("trl_bench.")


def test_tuta_table_native_override_writes_table_pkl_with_checkpoint(tmp_path):
    """build_extractor_command(runner_override=<native>) -> the native table
    runner with --input_dir/--output_path + --model_path + --device_id.

    The override targets the TABLE pickle directly (not the column pickle): the
    native runner's forward pass emits the {table_id, table_embedding:
    {cls_embedding, ...}, ...} schema run_task.py reads for embedding_type='cls'.
    """
    from trl_bench.registry import _TABLE_NATIVE_RUNNERS
    ckpt_root = tmp_path / "ckpts"
    _touch_ckpt(ckpt_root / "tuta" / "tuta.bin")
    csv_dir = tmp_path / "tables_all"
    table_pkl = tmp_path / "table" / "tuta" / "spider_join.pkl"
    cmd = build_extractor_command(
        model="tuta", dataset="spider_join",
        input_dir=csv_dir, output_path=table_pkl,
        checkpoint_root=ckpt_root,
        runner_override=_TABLE_NATIVE_RUNNERS["tuta"],
    )
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.models.tuta.generate_table_embeddings_native"]
    args = _args_dict_from_cmd(cmd)
    assert args["--input_dir"] == str(csv_dir)
    assert args["--output_path"] == str(table_pkl)
    assert args["--model_path"] == str(ckpt_root / "tuta" / "tuta.bin")
    assert args["--device_id"] == "0"     # cuda -> "0" via device_value_map
    assert "--device" not in args


def test_table_native_runners_emit_only_flags_their_runner_accepts(monkeypatch):
    """Every ``_TABLE_NATIVE_RUNNERS`` override emits only --flags its target
    runner's argparse declares (same invariant as the other Stage-1 surfaces).
    """
    import re
    import trl_bench
    from trl_bench.registry import build_extractor_command, _TABLE_NATIVE_RUNNERS
    monkeypatch.setattr(Path, "exists", lambda self: True)
    pkg_root = Path(trl_bench.__file__).resolve().parent

    def declared(runner):
        rel = runner.split(".", 1)[1].replace(".", "/") + ".py"
        return set(re.findall(r'add_argument\(\s*["\'](--[\w\-]+)',
                              (pkg_root / rel).read_text()))

    common = dict(dataset="ds", input_dir=Path("/in"),
                  output_path=Path("/out.pkl"), device="cuda",
                  checkpoint_root="/ck")
    for model, runner in _TABLE_NATIVE_RUNNERS.items():
        cmd = build_extractor_command(model=model, runner_override=runner, **common)
        emitted = [a for a in cmd[3:] if a.startswith("--")]
        miss = [f for f in emitted if f not in declared(runner)]
        assert not miss, (f"[table-native] {model}: emits {miss} not declared "
                          f"by {runner}")


def test_tabbie_extractor_command_uses_device_id_and_model_path(tmp_path):
    """TABBIE: --input / --output + --model_path <weights.pt> + --device_id 0."""
    ckpt_root = tmp_path / "ckpts"
    _touch_ckpt(ckpt_root / "tabbie" / "weights.pt")
    csv_dir = tmp_path / "tables_all"
    out_pkl = tmp_path / "tabbie_opendata.pkl"
    cmd = build_extractor_command(
        model="tabbie", dataset="opendata",
        input_dir=csv_dir, output_path=out_pkl,
        checkpoint_root=ckpt_root,
    )
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.models.tabbie.generate_column_embeddings"]
    args = _args_dict_from_cmd(cmd)
    assert args["--input"] == str(csv_dir)
    assert args["--output"] == str(out_pkl)
    assert args["--model_path"] == str(ckpt_root / "tabbie" / "weights.pt")
    assert args["--device_id"] == "0"
    assert "--device" not in args


def test_starmie_extractor_command_resolves_per_dataset_checkpoint(tmp_path):
    """Starmie: --input_dir / --output_path + --model_path <per-dataset .pt>.

    The checkpoint template uses ``{dataset}`` so two different datasets
    produce different resolved paths (each pointing at the per-dataset
    retrained binary under ``starmie/<dataset>/model_...pt``).
    """
    ckpt_root = tmp_path / "ckpts"
    _touch_ckpt(ckpt_root / "starmie" / "santos" / "datalake" /
                "model_drop_col,sample_row_head_column_0.pt")
    _touch_ckpt(ckpt_root / "starmie" / "tus" / "datalake" /
                "model_drop_col,sample_row_head_column_0.pt")
    csv_dir = tmp_path / "tables_all"

    cmd_santos = build_extractor_command(
        model="starmie", dataset="santos",
        input_dir=csv_dir, output_path=tmp_path / "santos.pkl",
        checkpoint_root=ckpt_root,
    )
    cmd_tus = build_extractor_command(
        model="starmie", dataset="tus",
        input_dir=csv_dir, output_path=tmp_path / "tus.pkl",
        checkpoint_root=ckpt_root,
    )
    assert cmd_santos[:3] == [sys.executable, "-m",
                              "trl_bench.models.starmie.generate_column_embeddings"]
    args_santos = _args_dict_from_cmd(cmd_santos)
    args_tus = _args_dict_from_cmd(cmd_tus)
    assert args_santos["--model_path"].endswith(
        "starmie/santos/datalake/model_drop_col,sample_row_head_column_0.pt"
    )
    assert args_tus["--model_path"].endswith(
        "starmie/tus/datalake/model_drop_col,sample_row_head_column_0.pt"
    )
    assert args_santos["--model_path"] != args_tus["--model_path"]
    # Starmie wrapper has no --device flag (device picked internally).
    assert "--device" not in args_santos
    assert "--device_id" not in args_santos


def test_extractor_missing_checkpoint_raises_clear_setting_error(tmp_path):
    """When ``checkpoint_required=True`` and the resolved file is missing,
    the dispatcher raises ``SettingError`` with the resolved path AND a
    pointer to ``scripts/download_checkpoints.sh``.
    """
    # No file touched under tmp_path/ckpts -> missing.
    ckpt_root = tmp_path / "ckpts"
    with pytest.raises(SettingError, match=r"checkpoint not found"):
        build_extractor_command(
            model="tabert", dataset="spider_join",
            input_dir=tmp_path / "tables_all",
            output_path=tmp_path / "out.pkl",
            checkpoint_root=ckpt_root,
        )
    # Same shape: the error mentions download_checkpoints.sh.
    with pytest.raises(SettingError, match=r"download_checkpoints\.sh"):
        build_extractor_command(
            model="tuta", dataset="opendata",
            input_dir=tmp_path / "tables_all",
            output_path=tmp_path / "out.pkl",
            checkpoint_root=ckpt_root,
        )


def test_extractor_checkpoint_root_defaults_to_checkpoints_subdir(tmp_path, monkeypatch):
    """When ``checkpoint_root`` is None, the dispatcher resolves against
    ``./checkpoints`` (relative to CWD). We chdir into ``tmp_path`` so the
    test is hermetic (and the missing-file guard trips because no
    ``checkpoints/tabert/...`` exists under ``tmp_path``).
    """
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SettingError, match=r"checkpoint not found"):
        build_extractor_command(
            model="tabert", dataset="spider_join",
            input_dir=tmp_path / "tables_all",
            output_path=tmp_path / "out.pkl",
        )
    # Now create the expected file under ``./checkpoints/`` and verify the
    # path resolves and the command builds cleanly.
    _touch_ckpt(tmp_path / "checkpoints" / "tabert" /
                "tabert_base_k3" / "model.bin")
    cmd = build_extractor_command(
        model="tabert", dataset="spider_join",
        input_dir=tmp_path / "tables_all",
        output_path=tmp_path / "out.pkl",
    )
    args = _args_dict_from_cmd(cmd)
    # Resolved path is the wholly-relative form, which str()-renders cleanly.
    assert args["--checkpoint"].endswith("tabert/tabert_base_k3/model.bin")


# == Table-direct models: not in _MODEL_EXTRACTORS (correct by design) ======
# mpnet, sentence_t5, tapex route through ``_TABLE_ENCODERS`` /
# ``build_table_encoder_command`` (table-DIRECT Stage-1; Stage-2 skipped).
# ``build_extractor_command`` is the column-extractor factory and MUST refuse
# them so callers know to use the table-encoder path instead. The per-wrapper
# table-encoder command-shape tests live below.

_TABLE_DIRECT_MODELS = ["mpnet", "sentence_t5", "tapex"]


@pytest.mark.parametrize("model", _TABLE_DIRECT_MODELS)
def test_table_direct_models_not_in_column_extractor_dispatch(model, tmp_path):
    """``build_extractor_command`` refuses table-direct models with a clear
    error naming ``_MODEL_EXTRACTORS`` -- callers should route via
    ``build_table_encoder_command`` instead. (The auto-orchestrator in
    ``run.py`` does this branch selection automatically.)
    """
    with pytest.raises(SettingError, match="_MODEL_EXTRACTORS"):
        build_extractor_command(
            model=model, dataset="spider_join",
            input_dir=tmp_path / "tables_all",
            output_path=tmp_path / "out.pkl",
        )


# == build_query_extractor_command: per-model command shape ==================
# Query-side Stage-1 (text encoder over questions JSON) — separate code path
# from the table-side column extractor. Wired wrappers: bert, gte, mpnet,
# sentence_t5, openai (all share the same CLI surface via
# ``generate_text_embeddings.py``; openai omits --device).


def test_mpnet_query_extractor_command_uses_question_fields(tmp_path):
    """mpnet query encoder runs with --mode cls + question/question_id fields."""
    in_json = tmp_path / "train.json"
    out_pkl = tmp_path / "queries_train.pkl"
    cmd = build_query_extractor_command(
        model="mpnet", input_json=in_json, output_path=out_pkl,
    )
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.models.mpnet.generate_text_embeddings"]
    args = dict(zip(cmd[3::2], cmd[4::2]))
    assert args["--mode"] == "cls"
    assert args["--input_json"] == str(in_json)
    assert args["--text_field"] == "question"
    assert args["--id_field"] == "question_id"
    assert args["--output"] == str(out_pkl)
    assert args["--device"] == "cuda"


def test_sentence_t5_query_extractor_command_uses_correct_runner(tmp_path):
    cmd = build_query_extractor_command(
        model="sentence_t5",
        input_json=tmp_path / "dev.json",
        output_path=tmp_path / "queries_dev.pkl",
    )
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.models.sentence_t5.generate_text_embeddings"]


def test_bert_query_extractor_command_supports_device_override(tmp_path):
    cmd = build_query_extractor_command(
        model="bert",
        input_json=tmp_path / "train.json",
        output_path=tmp_path / "queries_train.pkl",
        device="cpu",
    )
    args = dict(zip(cmd[3::2], cmd[4::2]))
    assert args["--device"] == "cpu"


def test_openai_query_extractor_command_omits_device_flag(tmp_path):
    """OpenAI client routes via HTTP and has no --device flag."""
    cmd = build_query_extractor_command(
        model="openai",
        input_json=tmp_path / "train.json",
        output_path=tmp_path / "queries_train.pkl",
    )
    assert "--device" not in cmd


def test_query_extractor_unwired_model_raises(tmp_path):
    """An unwired model triggers a clear SettingError naming the registry constant."""
    with pytest.raises(SettingError, match="_QUERY_ENCODER_EXTRACTORS"):
        build_query_extractor_command(
            model="tabicl",  # row-level wrapper, no query encoder
            input_json=tmp_path / "train.json",
            output_path=tmp_path / "queries_train.pkl",
        )


# == build_table_encoder_command: per-model command shape ===================
# Table-direct Stage-1: the runner writes a table-level pickle in one pass
# (no column pickle, no Stage-2 aggregator). Wired wrappers: mpnet,
# sentence_t5 (shared text-encoder runner) and tapex (its own runner).


def test_mpnet_table_encoder_command_has_expected_shape(tmp_path):
    """mpnet -> shared text-encoder runner + --pooling mean + --model
    sentence-transformers/all-mpnet-base-v2 + --input_dir / --output_path.
    """
    csv_dir = tmp_path / "tables_all"
    out_pkl = tmp_path / "embeddings" / "table" / "mpnet" / "nq_tables.pkl"
    cmd = build_table_encoder_command(
        model="mpnet", dataset="nq_tables",
        input_dir=csv_dir, output_path=out_pkl,
    )
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.utils.generate_table_embeddings_text_encoder"]
    args = _args_dict_from_cmd(cmd)
    assert args["--input_dir"] == str(csv_dir)
    assert args["--output_path"] == str(out_pkl)
    assert args["--model"] == "sentence-transformers/all-mpnet-base-v2"
    assert args["--pooling"] == "mean"
    # Table-direct path has no --device flag on this runner (device picked
    # internally via torch.cuda.is_available()).
    assert "--device" not in args
    # Must NOT emit --input / --output (column-extractor shape) by mistake.
    assert "--input" not in args
    assert "--output" not in args


def test_sentence_t5_table_encoder_command_has_expected_shape(tmp_path):
    """sentence_t5 -> shared text-encoder runner + --pooling mean +
    sentence-transformers/sentence-t5-base.
    """
    csv_dir = tmp_path / "tables_all"
    out_pkl = tmp_path / "sentence_t5_nq_tables.pkl"
    cmd = build_table_encoder_command(
        model="sentence_t5", dataset="nq_tables",
        input_dir=csv_dir, output_path=out_pkl,
    )
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.utils.generate_table_embeddings_text_encoder"]
    args = _args_dict_from_cmd(cmd)
    assert args["--input_dir"] == str(csv_dir)
    assert args["--output_path"] == str(out_pkl)
    assert args["--model"] == "sentence-transformers/sentence-t5-base"
    assert args["--pooling"] == "mean"


def test_tapex_table_encoder_command_has_expected_shape(tmp_path):
    """tapex -> dedicated runner + --model microsoft/tapex-base.

    TAPEX's wrapper has NO --pooling flag (pooling is hardcoded to
    mean-pool over non-padding encoder tokens); the dispatcher must
    therefore NOT emit --pooling.
    """
    csv_dir = tmp_path / "tables_all"
    out_pkl = tmp_path / "tapex_spider_join.pkl"
    cmd = build_table_encoder_command(
        model="tapex", dataset="spider_join",
        input_dir=csv_dir, output_path=out_pkl,
    )
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.models.tapex.generate_table_embeddings"]
    args = _args_dict_from_cmd(cmd)
    assert args["--input_dir"] == str(csv_dir)
    assert args["--output_path"] == str(out_pkl)
    assert args["--model"] == "microsoft/tapex-base"
    assert "--pooling" not in args      # TAPEX wrapper has no --pooling flag
    assert "--input" not in args
    assert "--output" not in args


def test_table_encoder_unwired_model_raises(tmp_path):
    """An unwired model triggers a clear SettingError naming the registry constant."""
    with pytest.raises(SettingError, match="_TABLE_ENCODERS"):
        build_table_encoder_command(
            model="bert",   # column extractor, not in _TABLE_ENCODERS
            dataset="spider_join",
            input_dir=tmp_path / "tables_all",
            output_path=tmp_path / "out.pkl",
        )


def test_table_encoder_runner_paths_are_dotted_modules():
    """Each runner is a dotted Python module path (no .py, lives under trl_bench)."""
    for model, cfg in _TABLE_ENCODERS.items():
        assert "/" not in cfg.runner, f"{model}: runner must be dotted"
        assert not cfg.runner.endswith(".py"), f"{model}: drop .py suffix"
        assert cfg.runner.startswith("trl_bench."), (
            f"{model}: runner must live under trl_bench.* for `python -m` resolution"
        )


def test_table_encoder_models_disjoint_from_column_extractors():
    """A model in ``_TABLE_ENCODERS`` must NOT also appear in ``_MODEL_EXTRACTORS``.

    The two dispatch paths are mutually exclusive: ``_TABLE_ENCODERS`` writes
    the table pickle directly (Stage-2 skipped), ``_MODEL_EXTRACTORS`` writes
    a column pickle that Stage-2 aggregates. ``_resolve_embeddings_path``
    selects the branch by membership; an overlap would create ambiguity.
    """
    overlap = set(_TABLE_ENCODERS) & set(_MODEL_EXTRACTORS)
    assert not overlap, (
        f"models in both _TABLE_ENCODERS and _MODEL_EXTRACTORS: {sorted(overlap)}"
    )


def test_table_encoder_models_appear_in_granularities():
    """Every table-encoder model must declare its granularity in ``_MODEL_GRANULARITIES``."""
    from trl_bench.registry import _MODEL_GRANULARITIES
    for model in _TABLE_ENCODERS:
        assert model in _MODEL_GRANULARITIES, (
            f"{model!r} wired in _TABLE_ENCODERS but missing from _MODEL_GRANULARITIES"
        )


def test_table_encoder_config_extra_args_is_tuple_of_pairs():
    """extra_args is tuple-of-tuples (immutable, frozen-dataclass-safe)."""
    for model, cfg in _TABLE_ENCODERS.items():
        assert isinstance(cfg.extra_args, tuple), f"{model}: extra_args must be tuple"
        for entry in cfg.extra_args:
            assert isinstance(entry, tuple) and len(entry) == 2, (
                f"{model}: each extra_args entry must be (flag, value)"
            )

def test_query_extractor_custom_text_field_propagates(tmp_path):
    """semantic_parsing-style use with --tokens_field-equivalent overrides."""
    cmd = build_query_extractor_command(
        model="mpnet",
        input_json=tmp_path / "in.json",
        output_path=tmp_path / "out.pkl",
        text_field="utterance",
        id_field="example_id",
    )
    args = dict(zip(cmd[3::2], cmd[4::2]))
    assert args["--text_field"] == "utterance"
    assert args["--id_field"] == "example_id"


# == starmie auto-pretrain wiring ===========================================

def test_starmie_checkpoint_path_matches_extractor_template():
    """The auto-pretrain hook must produce the EXACT .pt that
    ``build_extractor_command`` later resolves from
    ``ExtractorConfig.checkpoint_template`` -- otherwise extraction still fails
    after an hour of pretraining (the cross-code-path contract bug class)."""
    from trl_bench.registry import starmie_checkpoint_path, _MODEL_EXTRACTORS
    cfg = _MODEL_EXTRACTORS["starmie"]
    expected = Path("/ck") / cfg.checkpoint_template.format(dataset="valentine")
    assert starmie_checkpoint_path("valentine", "/ck") == expected


def test_build_starmie_pretrain_command_targets_dataset_checkpoint_dir():
    """run_pretrain writes ``<checkpoint_dir>/datalake/<model>.pt`` (verified
    against the santos checkpoint on disk), so the auto-pretrain command's
    ``--checkpoint_dir`` must be ``<ckpt_root>/starmie/<dataset>`` for the .pt
    to land where the extractor template resolves it."""
    from trl_bench.registry import build_starmie_pretrain_command
    cmd = build_starmie_pretrain_command(
        "valentine", Path("/data/valentine/tables"), "/ck")
    assert cmd[:3] == [sys.executable, "-m",
                       "trl_bench.models.starmie.run_pretrain"]
    assert cmd[cmd.index("--data_path") + 1] == "/data/valentine/tables"
    assert cmd[cmd.index("--checkpoint_dir") + 1] == "/ck/starmie/valentine"
    assert "--save_model" in cmd


def test_starmie_pretrain_output_dir_matches_extractor_checkpoint_dir(tmp_path):
    """The auto-pretrain command must write the checkpoint into the SAME
    directory the extractor later reads -- regardless of the input tables-dir
    name. Bug #7: run_pretrain derives the checkpoint subdir from
    ``basename(--data_path)`` (e.g. 'tables' for valentine, 'sato' for sato),
    but the extractor template hardcodes 'datalake'; a fresh pretrain on any
    non-'datalake' dataset landed the .pt where extraction could not find it
    (rc=1, no envelope). The build command must pin the subdir
    (``--checkpoint_subdir``) to the template's constant so producer and
    consumer agree for every dataset, not just union_search's datalake."""
    from trl_bench.registry import (
        build_starmie_pretrain_command, starmie_checkpoint_path,
    )
    # Input dir basename is 'tables' (NOT 'datalake') -- the bug case.
    cmd = build_starmie_pretrain_command(
        "valentine", tmp_path / "valentine" / "tables", tmp_path / "ck")
    ck_dir = Path(cmd[cmd.index("--checkpoint_dir") + 1])
    subdir = cmd[cmd.index("--checkpoint_subdir") + 1]
    produced_dir = ck_dir / subdir
    consumed_dir = starmie_checkpoint_path("valentine", tmp_path / "ck").parent
    assert produced_dir == consumed_dir, (
        f"pretrain writes to {produced_dir} but extractor reads {consumed_dir}")


def test_build_starmie_pretrain_command_matches_reference_hyperparameters():
    """starmie auto-pretrain must use the paper checkpoint's baked-in hp, or the
    embeddings are near-random. The reference .pt's hp (torch.load(ckpt)['hp'])
    is max_len=256, fp16=True, max_rows=1000 -- but run_pretrain DEFAULTS to
    max_len=128 (which truncates each serialized table to HALF, so the
    contrastive loss stalls ~4.5 and recall@gt collapses 0.76->0.09), fp16=False,
    max_rows=None. The build command must pin all three. Verified e2e: with these
    flags, valentine schema_matching recall@gt=0.77 (~paper 0.7637); the loss
    converges 0.45->0.25 instead of stalling at 4.5."""
    from trl_bench.registry import build_starmie_pretrain_command
    cmd = build_starmie_pretrain_command(
        "valentine", "/data/valentine/tables", "/ck")
    assert cmd[cmd.index("--max_len") + 1] == "256"
    assert cmd[cmd.index("--max_rows") + 1] == "1000"
    assert "--fp16" in cmd


# == DLTE orchestrator port =================================================

def test_dlte_run_all_orchestrator_targets_existing_scripts():
    """The ported DLTE orchestrator (run_all.py) imports cleanly and its
    SCRIPTS_DIR resolves to the real step8/9/10 stage scripts -- guards the path
    adaptation from the reference implementation (downstream_tasks/dlte/scripts ->
    src/trl_bench/tasks/dlte/scripts). The stages are CPU-only."""
    from trl_bench.tasks.dlte import run_all
    for s in ("step8_faiss_retrieval.py", "step9_column_alignment.py",
              "step10_row_matching.py"):
        assert (run_all.SCRIPTS_DIR / s).exists(), \
            f"orchestrator references a missing stage script: {s}"
    assert (run_all.PROJECT_ROOT / "src" / "trl_bench").is_dir(), \
        "PROJECT_ROOT must be the repo root (cwd + embeddings defaults depend on it)"


def test_dlte_orchestrator_threads_data_root_into_every_stage_command(
    monkeypatch, tmp_path,
):
    """Every DLTE stage command (step8/9/10) must carry ``--data_root``.

    Regression for the port bug (caught only by a real run, job 3845132's
    predecessor): the orchestrator never passed ``--data_root``, so each step
    defaulted DATASET_ROOT to a nonexistent ``<project_root>/datasets/dlte_v1``
    and every stage died at step8 with FileNotFound on ``query_tasks.jsonl``.
    The sibling import/path test above did NOT catch it because the path is
    only consumed at stage runtime -- so this asserts the *built command*, the
    layer where the bug actually lived.
    """
    from trl_bench.tasks.dlte import run_all

    captured: list = []
    monkeypatch.setattr(run_all, "_run", lambda cmd: captured.append(list(cmd)))
    monkeypatch.setattr(run_all, "RESULTS_BASE", tmp_path)
    data_root = tmp_path / "data" / "dlte_v1"
    monkeypatch.setattr(run_all, "DATA_ROOT", data_root)
    monkeypatch.setattr(run_all, "s1_done", lambda *a, **k: False)
    monkeypatch.setattr(run_all, "s2_done", lambda *a, **k: False)
    monkeypatch.setattr(run_all, "s3_done", lambda *a, **k: False)

    run_all.run_s1(("random", "column_mean"))
    run_all.run_s2(("random", "column_mean", None))
    run_all.run_s3(("random", "column_mean", None, ["random"]))

    assert len(captured) == 3, "expected one command per stage (1/2/3)"
    for cmd in captured:
        assert "--data_root" in cmd, f"stage command missing --data_root: {cmd}"
        assert cmd[cmd.index("--data_root") + 1] == str(data_root)


# == SSL/trained-row row_prediction wiring ==================================

def test_build_row_data_commands_trained_model_trains_then_generates():
    """SSL/trained-row models (scarf, dae, ...) need a train pass (writes a
    checkpoint) THEN a generate pass (loads it -> unified_row_embedding dir).
    row_prediction must run both before the probe -- previously these printed
    'not wired' and required the separate slurm pipeline."""
    from trl_bench.registry import build_row_data_commands
    cmds = build_row_data_commands("scarf", "/data/openml_3", "/ck/scarf", "/emb/scarf")
    assert len(cmds) == 2, "trained model must emit [train, generate]"
    train, gen = cmds
    assert "train_scarf.py" in train[1]
    assert train[train.index("--data_dir") + 1] == "/data/openml_3"
    assert train[train.index("--checkpoint_dir") + 1] == "/ck/scarf"
    assert "generate_embeddings.py" in gen[1]
    assert gen[gen.index("--embedding_dir") + 1] == "/emb/scarf"
    assert gen[gen.index("--checkpoint_dir") + 1] == "/ck/scarf"


def test_build_row_data_commands_pretrained_model_only_generates():
    """Pretrained row-data models (tabicl) skip training -- one generate pass,
    and tabicl's args mapping has no checkpoint_dir."""
    from trl_bench.registry import build_row_data_commands
    cmds = build_row_data_commands("tabicl", "/data/openml_3", "/ck/tabicl", "/emb/tabicl")
    assert len(cmds) == 1, "pretrained model emits [generate] only"
    gen = cmds[0]
    assert "generate_embeddings_train_test.py" in gen[1]
    assert gen[gen.index("--data_dir") + 1] == "/data/openml_3"
    assert gen[gen.index("--embedding_dir") + 1] == "/emb/tabicl"
    assert "--checkpoint_dir" not in gen


def test_build_row_data_commands_tabtransformer_passes_paper_emb_dim_512():
    """tabtransformer's row_prediction must pass --emb_dim 512.

    The paper's reference row_data sbatch trained with ``--emb_dim 512`` (and the
    REF metadata.json records embedding_dim=512). ``train_tabtransformer.py``
    defaults --emb_dim to 32, so the value MUST be supplied from
    ``row_data_models.yaml`` -- otherwise the per-feature embedding is 16x too
    small and openml_3 row_prediction accuracy lands ~6% below the paper
    (fresh 0.897 vs ref 0.959). Regression: the YAML carried emb_dim: 32."""
    from trl_bench.registry import build_row_data_commands
    cmds = build_row_data_commands("tabtransformer", "/data/openml_3", "/ck/tt", "/emb/tt")
    train = cmds[0]
    assert "train_tabtransformer.py" in train[1]
    assert train[train.index("--emb_dim") + 1] == "512"


def test_build_row_data_commands_tuta_emits_model_path_and_native_arg_names():
    """TUTA's row_prediction Stage-1 runner (generate_row_embeddings.py) requires
    --model_path (argparse required=True) and uses NATIVE arg names that differ
    from the generic row-data CLI: --dataset_dir / --output_dir (not --data_dir /
    --embedding_dir). row_data_models.yaml maps these + a `checkpoint:` key.

    Regression: build_row_data_commands listed model_path in `skip` and never
    read the YAML `checkpoint:` key, so it DROPPED --model_path -> the runner
    failed with a missing-required-arg error. The canonical slurm generator
    (slurm/generate_row_data_scripts.py:115-118) DOES emit it; this is the
    faithful-port fix."""
    from trl_bench.registry import build_row_data_commands
    cmds = build_row_data_commands(
        "tuta", "/data/openml_3", "/ck/tuta", "/emb/tuta",
        checkpoint_root="/ckroot",
    )
    assert len(cmds) == 1, "tuta is pretrained -> single generate pass"
    gen = cmds[0]
    assert "generate_row_embeddings.py" in gen[1]
    assert gen[gen.index("--dataset_dir") + 1] == "/data/openml_3"
    assert gen[gen.index("--output_dir") + 1] == "/emb/tuta"
    # checkpoint resolves under the passed checkpoint_root (leading
    # "checkpoints/" stripped from the YAML value), matching how the rest of
    # run.py resolves licensed checkpoints (<ckpt_root>/<template>).
    assert gen[gen.index("--model_path") + 1] == "/ckroot/tuta/tuta.bin"
    assert gen[gen.index("--label_policy") + 1] == "manifest"
    assert gen[gen.index("--model_type") + 1] == "tuta"
    # The generic --data_dir/--embedding_dir flags must NOT appear (tuta uses
    # its native names).
    assert "--data_dir" not in gen
    assert "--embedding_dir" not in gen


def test_build_row_data_commands_tabbie_emits_model_path():
    """TABBIE's row_prediction runner (generate_embeddings_train_test.py) requires
    --model_path (argparse required=True). It uses the generic --data_dir /
    --embedding_dir names but ALSO needs the checkpoint. Regression: the
    checkpoint was dropped -> runner failed."""
    from trl_bench.registry import build_row_data_commands
    cmds = build_row_data_commands(
        "tabbie", "/data/openml_3", "/ck/tabbie", "/emb/tabbie",
        checkpoint_root="/ckroot",
    )
    assert len(cmds) == 1, "tabbie is pretrained -> single generate pass"
    gen = cmds[0]
    assert "generate_embeddings_train_test.py" in gen[1]
    assert gen[gen.index("--data_dir") + 1] == "/data/openml_3"
    assert gen[gen.index("--embedding_dir") + 1] == "/emb/tabbie"
    assert gen[gen.index("--model_path") + 1] == "/ckroot/tabbie/weights.pt"
    assert gen[gen.index("--label_policy") + 1] == "manifest"


def test_build_row_data_commands_hf_and_pretrained_no_checkpoint_omit_model_path():
    """HF text encoders (bert/gte) and checkpoint-less pretrained models (tabicl)
    have NO `checkpoint:` key in row_data_models.yaml, so --model_path must NOT
    be emitted even when a checkpoint_root is passed. Their runners resolve the
    HF model id from --model (a default flag), not a local checkpoint."""
    from trl_bench.registry import build_row_data_commands
    for model in ("bert", "gte", "tabicl"):
        cmds = build_row_data_commands(
            model, "/data/openml_3", "/ck/x", "/emb/x",
            checkpoint_root="/ckroot",
        )
        gen = cmds[-1]
        assert "--model_path" not in gen, (
            f"{model} has no checkpoint: key -> must not emit --model_path"
        )
    # bert/gte also carry their HF model id + tokenization defaults from YAML.
    bert = build_row_data_commands("bert", "/d", "/c", "/e")[-1]
    assert bert[bert.index("--model") + 1] == "bert-base-uncased"
    assert bert[bert.index("--max_length") + 1] == "512"
    gte = build_row_data_commands("gte", "/d", "/c", "/e")[-1]
    assert gte[gte.index("--model") + 1] == "thenlper/gte-base"


def test_build_row_data_commands_trained_model_gets_no_model_path():
    """Trained SSL models (scarf) have no `checkpoint:` key and no model_path in
    their args map, so neither the train nor the generate command may gain a
    --model_path under the checkpoint fix. They keep using --checkpoint_dir for
    the per-dataset training artifact."""
    from trl_bench.registry import build_row_data_commands
    cmds = build_row_data_commands(
        "scarf", "/data/openml_3", "/ck/scarf", "/emb/scarf",
        checkpoint_root="/ckroot",
    )
    assert len(cmds) == 2
    for c in cmds:
        assert "--model_path" not in c
    train, gen = cmds
    assert train[train.index("--checkpoint_dir") + 1] == "/ck/scarf"
    assert gen[gen.index("--checkpoint_dir") + 1] == "/ck/scarf"


def test_row_runner_models_are_declared_row_capable():
    """Every model wired as a row runner -- in _ROW_RUNNERS (record_linkage) or
    _ROW_DATA_RUNNERS (row_prediction) -- must declare "row" in its granularity,
    else run.py's auto-extract rejects it as "not row-capable" (run.py:1047).

    Regression: openai was wired in BOTH row maps but declared {col, table};
    the reference masked it by pre-extracting embeddings (--embeddings-path), so
    a fresh `run.py --model openai --task record_linkage` failed."""
    from trl_bench.registry import (
        _MODEL_GRANULARITIES, _ROW_RUNNERS, _ROW_DATA_RUNNERS,
    )
    for model in sorted(set(_ROW_RUNNERS) | set(_ROW_DATA_RUNNERS)):
        grans = _MODEL_GRANULARITIES.get(model, frozenset())
        assert "row" in grans, (
            f"{model} is a wired row runner but its granularity is "
            f"{sorted(grans)} (missing 'row') -> run.py rejects its row "
            f"auto-extract as not row-capable"
        )
