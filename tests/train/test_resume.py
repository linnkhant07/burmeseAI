from burmeseai.train import (
    RESUME_RETRY_CAP,
    latest_checkpoint,
    plan_resume,
    read_checkpoint,
    train_on_volume,
    write_checkpoint,
)


def test_latest_checkpoint_is_the_highest_step(tmp_path):
    write_checkpoint(tmp_path, 2)
    write_checkpoint(tmp_path, 10)
    write_checkpoint(tmp_path, 3)
    (tmp_path / "checkpoint-notes.json").write_text("{}", encoding="utf-8")
    found = latest_checkpoint(tmp_path)
    assert found is not None
    assert found.name == "checkpoint-000010.json"
    assert read_checkpoint(found)["step"] == 10


def test_first_attempt_starts_and_a_resume_loads_the_latest(tmp_path):
    first = train_on_volume(tmp_path)
    assert first["attempt"] == 0
    assert first["action"] == "start"
    assert first["training_started"] is False
    write_checkpoint(tmp_path, 4, {"marker": "partial"})
    write_checkpoint(tmp_path, 11, {"marker": "latest"})
    second = train_on_volume(tmp_path)
    assert second["attempt"] == 1
    assert second["action"] == "resume"
    assert second["resumed_step"] == 11
    assert second["checkpoint"] == "checkpoint-000011.json"
    assert second["training_started"] is False


def test_resume_without_a_checkpoint_does_not_start_over(tmp_path):
    train_on_volume(tmp_path)
    second = train_on_volume(tmp_path)
    assert second["attempt"] == 1
    assert second["action"] == "stop"
    assert second["checkpoint"] is None
    assert second["training_started"] is False
    assert "step zero" in second["reason"]


def test_retry_cap_is_one_resume(tmp_path):
    assert RESUME_RETRY_CAP == 1
    train_on_volume(tmp_path)
    write_checkpoint(tmp_path, 1)
    assert train_on_volume(tmp_path)["action"] == "resume"
    third = train_on_volume(tmp_path)
    assert third["attempt"] == 2
    assert third["action"] == "stop"
    assert plan_resume(2, latest_checkpoint(tmp_path), cap=1) == "stop"
    assert plan_resume(1, None, cap=1) == "stop"
    assert plan_resume(0, None, cap=1) == "start"
    assert plan_resume(0, latest_checkpoint(tmp_path), cap=1) == "resume"
