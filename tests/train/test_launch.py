import os
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

from burmeseai.train import (
    estimated_cost_usd,
    l4_resources,
    lora_enabled,
    main,
    missing_token_message,
    next_batch_size,
    run_gated_launch,
)

ROOT = Path(__file__).resolve().parents[2]


def test_missing_tokens_exit_2_and_do_not_launch(tmp_path, capsys):
    calls = {"budget": 0, "launch": 0}

    def budget():
        calls["budget"] += 1
        return True

    def launch(resources):
        calls["launch"] += 1
        del resources

    code = run_gated_launch(
        environ={},
        fertility_path=tmp_path / "fertility.json",
        budget_set=budget,
        launch=launch,
    )
    assert code == 2
    assert calls == {"budget": 0, "launch": 0}
    assert "not requesting a GPU" in capsys.readouterr().out

    only_secret = run_gated_launch(
        environ={"MODAL_TOKEN_SECRET": "secret"},
        fertility_path=tmp_path / "fertility.json",
        budget_set=budget,
        launch=launch,
    )
    assert only_secret == 2
    assert missing_token_message(["MODAL_TOKEN_ID"]) == "MODAL_TOKEN_ID is missing; not requesting a GPU"
    blank = run_gated_launch(
        environ={"MODAL_TOKEN_ID": "", "MODAL_TOKEN_SECRET": "secret"},
        fertility_path=tmp_path / "fertility.json",
        budget_set=budget,
        launch=launch,
    )
    assert blank == 2
    assert calls == {"budget": 0, "launch": 0}


def test_missing_fertility_does_not_start_a_gpu(tmp_path, capsys):
    launched = []
    budgets = []
    code = run_gated_launch(
        environ={"MODAL_TOKEN_ID": "id", "MODAL_TOKEN_SECRET": "secret"},
        fertility_path=tmp_path / "fertility.json",
        budget_set=lambda: budgets.append(True) or True,
        launch=launched.append,
    )
    assert code == 1
    assert launched == []
    assert budgets == []
    assert "not starting a GPU" in capsys.readouterr().out


def test_budget_failure_does_not_start_a_gpu(tmp_path, capsys):
    fertility = tmp_path / "fertility.json"
    fertility.write_text("{}\n", encoding="utf-8")
    launched = []
    code = run_gated_launch(
        environ={"MODAL_TOKEN_ID": "id", "MODAL_TOKEN_SECRET": "secret"},
        fertility_path=fertility,
        budget_set=lambda: False,
        launch=launched.append,
    )
    assert code == 1
    assert launched == []
    assert "workspace usage budget of $20 cannot be set" in capsys.readouterr().out


def test_open_gates_request_one_l4(tmp_path):
    fertility = tmp_path / "fertility.json"
    fertility.write_text("{}\n", encoding="utf-8")
    launched = []
    code = run_gated_launch(
        environ={"MODAL_TOKEN_ID": "id", "MODAL_TOKEN_SECRET": "secret"},
        fertility_path=fertility,
        budget_set=lambda: True,
        launch=launched.append,
    )
    assert code == 0
    assert launched == [l4_resources()]
    resources = launched[0]
    assert resources["gpu"] == "L4"
    assert resources["cpu"] == 2
    assert resources["memory_mib"] == 16384
    assert resources["max_resume_retries"] == 1
    assert resources["region"] is None
    assert resources["nonpreemptible"] is False
    assert resources["timeout_seconds"] == 18 * 60 * 60


def test_module_main_without_smoke_exits_2_when_tokens_are_absent(monkeypatch, capsys):
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    assert main([]) == 2
    assert "not requesting a GPU" in capsys.readouterr().out


def test_oom_halves_the_batch_and_lora_waits_for_a_source():
    assert next_batch_size(4, out_of_memory=True) == 2
    assert next_batch_size(1, out_of_memory=True) == 1
    assert next_batch_size(4, out_of_memory=False) == 4
    assert lora_enabled(0) is False
    assert lora_enabled(1) is True


def test_eighteen_hour_rate_card():
    assert estimated_cost_usd(18 * 60 * 60) == pytest.approx(18.385056)


def _block_network():
    original_connect = socket.socket.connect
    original_create = socket.create_connection

    def _refuse(*args, **kwargs):
        raise RuntimeError("network")

    socket.socket.connect = _refuse
    socket.create_connection = _refuse
    return original_connect, original_create


def _restore_network(original_connect, original_create):
    socket.socket.connect = original_connect
    socket.create_connection = original_create


def test_workspace_budget_cannot_be_set_without_a_network():
    pytest.importorskip("modal")
    calls = []
    original_connect = socket.socket.connect
    original_create = socket.create_connection

    def _refuse(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("network")

    socket.socket.connect = _refuse
    socket.create_connection = _refuse
    try:
        from burmeseai.train import workspace_usage_budget_can_be_set

        assert workspace_usage_budget_can_be_set() is False
        assert calls == []
    finally:
        _restore_network(original_connect, original_create)


def test_registered_function_is_one_l4_without_a_region_pin():
    pytest.importorskip("modal")
    original = _block_network()
    try:
        from burmeseai import train as train_module

        app = train_module.build_modal_app()
        raw = next(
            value for value in train_module._L4.__dict__.values() if hasattr(value, "_function_info")
        )
        info = raw._function_info
        assert info.gpus == [("L4", 1)]
        assert info.cpu == 2
        assert info.memory_mib == 16384
        assert info.timeout == 18 * 60 * 60
        assert info.max_retries == 1
        assert info.nonpreemptible is False
        assert info.regions is None
        assert raw._spec.scheduler_placement is None
        assert list(raw._spec.volumes) == ["/checkpoints"]
        assert app.name == "burmeseai-train"
    finally:
        _restore_network(*original)


def test_modal_run_exits_2_when_tokens_are_missing():
    modal_bin = shutil.which("modal")
    if modal_bin is None:
        candidate = Path.home() / ".local" / "bin" / "modal"
        if candidate.is_file():
            modal_bin = str(candidate)
    if modal_bin is None:
        pytest.skip("modal CLI is not installed")
    env = os.environ.copy()
    env.pop("MODAL_TOKEN_ID", None)
    env.pop("MODAL_TOKEN_SECRET", None)
    completed = subprocess.run(
        [modal_bin, "run", "burmeseai/train.py"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "MODAL_TOKEN_ID or MODAL_TOKEN_SECRET is missing" in completed.stdout
    assert "not requesting a GPU" in completed.stdout
    combined = completed.stdout + completed.stderr
    assert "GPU training" not in combined
