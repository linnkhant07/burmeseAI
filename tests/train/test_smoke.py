import os
import subprocess
import sys
from pathlib import Path

from burmeseai.train import (
    BATCH_SIZE,
    load_fixture,
    mean_init_resize,
    mean_init_row,
    run_smoke,
    sequence_loss_and_grad,
)

ROOT = Path(__file__).resolve().parents[2]


def test_gradient_matches_a_finite_difference():
    embedding = [
        [0.2, -0.1],
        [0.0, 0.3],
        [-0.2, 0.1],
    ]
    sequences = [[0, 1, 2]]
    _, grad = sequence_loss_and_grad(embedding, sequences)
    epsilon = 1e-5
    for row_index, row in enumerate(embedding):
        for dim_index in range(len(row)):
            up = [list(values) for values in embedding]
            down = [list(values) for values in embedding]
            up[row_index][dim_index] += epsilon
            down[row_index][dim_index] -= epsilon
            loss_up, _ = sequence_loss_and_grad(up, sequences)
            loss_down, _ = sequence_loss_and_grad(down, sequences)
            numeric = (loss_up - loss_down) / (2 * epsilon)
            assert abs(numeric - grad[row_index][dim_index]) < 1e-6


def test_smoke_reads_the_tie_flag_and_takes_two_steps(capsys):
    loaded = load_fixture()
    assert loaded["tie_word_embeddings"] is True
    assert loaded["input_embeddings"] is loaded["output_embeddings"]
    assert len(loaded["sequences"]) == BATCH_SIZE
    result = run_smoke()
    captured = capsys.readouterr()
    assert result["tie_word_embeddings"] is True
    assert result["same_matrix"] is True
    assert len(result["losses"]) == 2
    assert result["vocab_after"] == result["vocab_before"] + 2
    assert all(loss > 0 for loss in result["losses"])
    expected = [
        mean_init_row(loaded["input_embeddings"], pieces)
        for pieces in loaded["new_token_pieces"]
    ]
    for got, want in zip(result["init_rows"], expected):
        assert got == want
    fresh, fresh_output = mean_init_resize(
        loaded["input_embeddings"],
        loaded["new_token_pieces"],
        loaded["input_embeddings"],
        tie_word_embeddings=True,
    )
    assert fresh_output is fresh
    assert result["embedding"] != fresh
    assert result["losses"][0] != result["losses"][1]
    assert "steps: 2" in captured.out
    assert "mean_init: per-piece" in captured.out
    assert "device: cpu" in captured.out
    assert "modal_imported: false" in captured.out


def test_smoke_process_does_not_import_modal_or_open_a_socket():
    env = os.environ.copy()
    env["MODAL_TOKEN_ID"] = "should-not-appear"
    env["MODAL_TOKEN_SECRET"] = "also-should-not-appear"
    env.pop("PYTHONPATH", None)
    script = """
import socket
import sys

def _refuse(*args, **kwargs):
    raise RuntimeError("network")

socket.socket.connect = _refuse
socket.create_connection = _refuse

sys.path.insert(0, %r)
sys.argv = ["burmeseai.train", "--smoke"]
import runpy
runpy.run_module("burmeseai.train", run_name="__main__")
""" % str(ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "should-not-appear" not in completed.stdout
    assert "also-should-not-appear" not in completed.stdout
    assert "modal_imported: false" in completed.stdout
    assert "steps: 2" in completed.stdout
    assert "loss: " in completed.stdout
