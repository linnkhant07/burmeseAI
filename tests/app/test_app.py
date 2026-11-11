"""CLI tests. They do not download weights and do not open a network connection."""

from __future__ import annotations

import builtins
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from burmeseai import app
from burmeseai.retrieve import Hit, Index, load_chunks

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("safety_fixture.json")


def fixture_index() -> Index:
    return Index(load_chunks(str(FIXTURE)))


class RouterTests(unittest.TestCase):
    def test_keyword_and_script_routes(self) -> None:
        cases = [
            ("hello there", "chat", None),
            ("theater seats", "chat", None),
            ("cheat code", "chat", None),
            ("earthquake nearby", "safety", None),
            ("what is the heat like", "safety", None),
            ("health question", "safety", None),
            ("aftershock", "safety", None),
            ("ငလျင်", "safety", None),
            ("အပူလှိုင်း", "safety", None),
            ("ကျန်းမာရေး", "safety", None),
            ("translate good morning to burmese", "translation", "en-my"),
            ("translate this to english", "translation", "my-en"),
            ("english to burmese please", "translation", "en-my"),
            ("ဘာသာပြန်", "translation", "my-en"),
            ("မြန်မာလို", "translation", "en-my"),
            ("အင်္ဂလိပ်လို", "translation", "my-en"),
        ]
        for text, lane, direction in cases:
            with self.subTest(text=text):
                decision = app.route(text)
                self.assertEqual(decision.lane, lane)
                self.assertEqual(decision.direction, direction)

    def test_safety_wins_over_translation(self) -> None:
        decision = app.route("translate the earthquake note to english")
        self.assertEqual(decision.lane, "safety")
        self.assertIsNone(decision.direction)

    def test_fake_converter_is_what_the_router_sees(self) -> None:
        refused = app.respond(
            "xxxx",
            index=Index(),
            converter=lambda _text: "earthquake nearby",
        )
        self.assertEqual(refused.text, "earthquake nearby")
        self.assertEqual(refused.lane, "safety")
        self.assertTrue(refused.refused)
        self.assertEqual(refused.reason, "empty_index")

        chat = app.respond(
            "earthquake",
            index=Index(),
            converter=lambda _text: "hello there friend",
        )
        self.assertEqual(chat.lane, "chat")
        self.assertFalse(chat.refused)

    def test_convert_mixed_is_called_when_the_module_exists(self) -> None:
        calls = []

        def convert_mixed(text: str) -> str:
            calls.append(text)
            return "heat nearby"

        module = type(sys)("burmeseai.zawgyi")
        module.convert_mixed = convert_mixed
        sys.modules.pop("burmeseai.zawgyi", None)
        with mock.patch.dict(sys.modules, {"burmeseai.zawgyi": module}):
            conversion = app.convert_text("original line")
        self.assertEqual(calls, ["original line"])
        self.assertEqual(conversion.original, "original line")
        self.assertEqual(conversion.unicode, "heat nearby")
        self.assertTrue(conversion.converted)

    def test_converter_runtime_failure_keeps_the_original_and_refuses(self) -> None:
        def broken(_text: str) -> str:
            raise RuntimeError("convertLine requires Node.js 20")

        with mock.patch.object(app, "load_convert_mixed", return_value=broken):
            reply = app.respond("earthquake nearby", index=Index())
        self.assertFalse(reply.converted)
        self.assertEqual(reply.text, "earthquake nearby")
        self.assertTrue(reply.refused)
        self.assertEqual(reply.reply, f"{app.MODEL_NOT_LOADED}\n{app.DISCLAIMER}")

    def test_missing_zawgyi_import_is_documented_and_keeps_the_original(self) -> None:
        self.assertIn("burmeseai.zawgyi.convert_mixed", app.load_convert_mixed.__doc__)
        self.assertIn("ImportError", app.load_convert_mixed.__doc__)
        real_import = builtins.__import__

        def guarded(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "burmeseai.zawgyi":
                raise ImportError("burmeseai.zawgyi is not available")
            return real_import(name, globals, locals, fromlist, level)

        sys.modules.pop("burmeseai.zawgyi", None)
        package = sys.modules.get("burmeseai")
        if package is not None and hasattr(package, "zawgyi"):
            delattr(package, "zawgyi")
        with mock.patch("builtins.__import__", guarded):
            conversion = app.convert_text("keep me")
        self.assertFalse(conversion.converted)
        self.assertEqual(conversion.original, "keep me")
        self.assertEqual(conversion.unicode, "keep me")


class ReplyTests(unittest.TestCase):
    def test_empty_index_refuses_with_the_disclaimer(self) -> None:
        reply = app.respond("earthquake nearby", index=Index())
        self.assertEqual(reply.lane, "safety")
        self.assertTrue(reply.refused)
        self.assertEqual(reply.reason, "empty_index")
        self.assertEqual(reply.hits, [])
        self.assertFalse(reply.model_loaded)
        self.assertFalse(reply.model_called)
        self.assertFalse(reply.enable_thinking)
        self.assertEqual(
            reply.reply,
            f"{app.MODEL_NOT_LOADED}\n{app.DISCLAIMER}",
        )
        self.assertEqual(reply.resident_memory, "not measured")
        self.assertEqual(reply.tokens_per_second, "not measured")
        with self.assertRaises(ValueError):
            float(reply.resident_memory)

    def test_low_score_does_not_invent_medical_text(self) -> None:
        reply = app.respond("medicine", index=fixture_index())
        self.assertEqual(reply.lane, "safety")
        self.assertTrue(reply.refused)
        self.assertEqual(reply.reason, "low_score")
        self.assertEqual(reply.hits, [])
        self.assertEqual(reply.reply, f"{app.MODEL_NOT_LOADED}\n{app.DISCLAIMER}")
        self.assertNotIn("wellness", reply.reply)
        self.assertNotIn("Earthquake", reply.reply)
        for banned in ("drink", "dosage", "cpr", "tablet", "shelter"):
            self.assertNotIn(banned, reply.reply.lower())

    def test_hit_quotes_the_chunk_and_does_not_call_the_model(self) -> None:
        def explode(*_args, **_kwargs):
            raise AssertionError("safety hit must not call the model")

        reply = app.respond("earthquake", index=fixture_index(), generate=explode)
        self.assertFalse(reply.refused)
        self.assertEqual(reply.reason, None)
        self.assertEqual([hit.chunk_id for hit in reply.hits], ["fixture-earthquake-1"])
        self.assertIn("[fixture-earthquake-1 |", reply.reply)
        self.assertIn("Earthquake quake seismic marker. Not an instruction.", reply.reply)
        self.assertIn(app.MODEL_NOT_LOADED, reply.reply)
        self.assertIn(app.DISCLAIMER, reply.reply)
        self.assertFalse(reply.model_called)
        self.assertEqual(reply.resident_memory, "not measured")
        self.assertEqual(reply.tokens_per_second, "not measured")

    def test_loaded_model_safety_refusal_is_only_the_disclaimer(self) -> None:
        hit = Hit(
            chunk_id="c",
            text="quoted span",
            license="CC0-1.0",
            score=2.0,
            topic="earthquake",
            source="project-written",
        )
        refused = app.compose_reply(
            lane="safety",
            refused=True,
            model_loaded=True,
            hits=[hit],
            generated="invented instruction",
        )
        self.assertEqual(refused, app.DISCLAIMER)
        quoted = app.compose_reply(
            lane="safety",
            refused=False,
            model_loaded=True,
            hits=[hit],
            generated="invented instruction",
        )
        self.assertEqual(quoted, f"[c | CC0-1.0] quoted span\n{app.DISCLAIMER}")
        self.assertNotIn("invented", quoted)

    def test_chat_and_translation_without_weights_do_not_fabricate_a_reply(self) -> None:
        chat = app.respond("hello there", index=fixture_index())
        self.assertEqual(chat.lane, "chat")
        self.assertEqual(chat.reply, app.MODEL_NOT_LOADED)
        self.assertNotIn(app.DISCLAIMER, chat.reply)
        self.assertFalse(chat.model_loaded)
        translation = app.respond("translate good morning to burmese")
        self.assertEqual(translation.lane, "translation")
        self.assertEqual(translation.direction, "en-my")
        self.assertEqual(translation.reply, app.MODEL_NOT_LOADED)
        self.assertNotIn("မင်္ဂလာ", translation.reply)

    def test_chat_calls_the_model_with_thinking_off_and_converted_text(self) -> None:
        seen = {}

        def generate(messages, *, enable_thinking=False):
            seen["messages"] = messages
            seen["thinking"] = enable_thinking
            return app.Generation(text="placeholder reply")

        reply = app.respond(
            "ZAWGYI",
            converter=lambda _text: "hello there friend",
            generate=generate,
        )
        self.assertEqual(reply.lane, "chat")
        self.assertTrue(reply.model_called)
        self.assertTrue(reply.model_loaded)
        self.assertFalse(seen["thinking"])
        self.assertFalse(reply.enable_thinking)
        self.assertEqual(seen["messages"][-1]["content"], "hello there friend")
        self.assertEqual(reply.reply, "placeholder reply")
        self.assertEqual(reply.resident_memory, "not measured")
        self.assertEqual(reply.tokens_per_second, "not measured")

    def test_default_generate_rejects_thinking_and_missing_weights(self) -> None:
        with self.assertRaises(ValueError):
            app.default_generate([], None, enable_thinking=True)
        self.assertIsNone(app.default_generate([{"role": "user", "content": "hi"}], None))
        self.assertIsNone(
            app.default_generate(
                [{"role": "user", "content": "hi"}],
                "/tmp/burmeseai-weights-not-present",
            )
        )

    def test_empty_weights_directory_still_refuses_safely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reply = app.respond("medicine", index=Index(), weights=tmp)
        self.assertTrue(reply.refused)
        self.assertEqual(reply.reason, "empty_index")
        self.assertFalse(reply.model_loaded)
        self.assertEqual(reply.reply, f"{app.MODEL_NOT_LOADED}\n{app.DISCLAIMER}")
        self.assertEqual(reply.resident_memory, "not measured")
        self.assertEqual(reply.tokens_per_second, "not measured")

    def test_status_has_no_fabricated_figures(self) -> None:
        payload = app.status_payload(None)
        self.assertEqual(payload["model"], "Qwen/Qwen3-0.6B")
        self.assertFalse(payload["model_loaded"])
        self.assertFalse(payload["enable_thinking"])
        self.assertEqual(payload["resident_memory"], "not measured")
        self.assertEqual(payload["tokens_per_second"], "not measured")
        self.assertEqual(payload["model_note"], app.MODEL_NOT_LOADED)


class CliTests(unittest.TestCase):
    def test_cli_empty_index_json(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "burmeseai.app", "--json", "earthquake nearby"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["lane"], "safety")
        self.assertTrue(payload["refused"])
        self.assertEqual(payload["reason"], "empty_index")
        self.assertEqual(payload["reply"], f"{app.MODEL_NOT_LOADED}\n{app.DISCLAIMER}")
        self.assertEqual(payload["resident_memory"], "not measured")
        self.assertEqual(payload["tokens_per_second"], "not measured")
        self.assertFalse(payload["enable_thinking"])
        self.assertFalse(payload["model_loaded"])
        self.assertEqual(payload["hits"], [])

    def test_cli_fixture_hit_and_low_score(self) -> None:
        hit = subprocess.run(
            [
                sys.executable,
                "-m",
                "burmeseai.app",
                "--json",
                "--index",
                str(FIXTURE),
                "earthquake",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(hit.returncode, 0, hit.stderr)
        payload = json.loads(hit.stdout)
        self.assertFalse(payload["refused"])
        self.assertEqual(payload["hits"][0]["id"], "fixture-earthquake-1")
        self.assertIn("Not an instruction.", payload["reply"])
        self.assertIn(app.DISCLAIMER, payload["reply"])

        miss = subprocess.run(
            [
                sys.executable,
                "-m",
                "burmeseai.app",
                "--json",
                "--index",
                str(FIXTURE),
                "aftershock",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(miss.returncode, 0, miss.stderr)
        missed = json.loads(miss.stdout)
        self.assertTrue(missed["refused"])
        self.assertEqual(missed["reason"], "low_score")
        self.assertEqual(missed["hits"], [])
        self.assertNotIn("Earthquake quake", missed["reply"])

    def test_cli_status_and_bad_index(self) -> None:
        status = subprocess.run(
            [sys.executable, "-m", "burmeseai.app", "--status", "--json"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(status.returncode, 0, status.stderr)
        payload = json.loads(status.stdout)
        self.assertEqual(payload["resident_memory"], "not measured")
        self.assertEqual(payload["tokens_per_second"], "not measured")
        self.assertFalse(payload["model_loaded"])

        bad = subprocess.run(
            [sys.executable, "-m", "burmeseai.app", "--index", str(ROOT / "missing-index.json"), "hi"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(bad.returncode, 2)
        self.assertIn("Could not read index", bad.stderr)

    def test_cli_stdin(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "burmeseai.app", "--json"],
            cwd=ROOT,
            input="translate good morning to burmese\n",
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["lane"], "translation")
        self.assertEqual(payload["direction"], "en-my")
        self.assertEqual(payload["reply"], app.MODEL_NOT_LOADED)

    def test_library_text_has_no_invented_procedure(self) -> None:
        for relative in ("burmeseai/retrieve.py", "burmeseai/app.py"):
            text = (ROOT / relative).read_text(encoding="utf-8").lower()
            for banned in ("drop, cover", "cpr", "dosage", "drink water", "first aid"):
                self.assertNotIn(banned, text)
            self.assertNotIn("tokens per second:", text)


if __name__ == "__main__":
    unittest.main()
