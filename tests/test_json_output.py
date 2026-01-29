"""Tests for JSON output functionality."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from claude_code_transcripts import (
    cli,
    generate_json_output,
    parse_session_file,
)


class TestGenerateJsonOutput:
    """Tests for the generate_json_output function."""

    def test_returns_dict_with_required_keys(self, tmp_path):
        """Test that generate_json_output returns dict with required top-level keys."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"type":"user","timestamp":"2025-01-01T10:00:00.000Z","message":{"role":"user","content":"Hello"}}\n'
            '{"type":"assistant","timestamp":"2025-01-01T10:00:05.000Z","message":{"role":"assistant","content":[{"type":"text","text":"Hi!"}]}}\n'
        )

        result = generate_json_output(jsonl_file)

        assert "metadata" in result
        assert "stats" in result
        assert "conversations" in result
        assert "commits" in result

    def test_metadata_has_generated_at(self, tmp_path):
        """Test that metadata includes generated_at timestamp."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"type":"user","timestamp":"2025-01-01T10:00:00.000Z","message":{"role":"user","content":"Hello"}}\n'
        )

        result = generate_json_output(jsonl_file)

        assert "generated_at" in result["metadata"]
        # Should be a valid ISO-8601 timestamp
        datetime.fromisoformat(
            result["metadata"]["generated_at"].replace("Z", "+00:00")
        )

    def test_metadata_includes_github_repo_when_detected(self, tmp_path):
        """Test that metadata includes github_repo when auto-detected."""
        jsonl_file = tmp_path / "test.jsonl"
        # Include a tool result with git push output
        jsonl_file.write_text(
            '{"type":"user","timestamp":"2025-01-01T10:00:00.000Z","message":{"role":"user","content":"Push changes"}}\n'
            '{"type":"assistant","timestamp":"2025-01-01T10:00:05.000Z","message":{"role":"assistant","content":[{"type":"tool_result","content":"remote: Create a pull request on GitHub by visiting:\\nremote:      https://github.com/owner/repo/pull/new/branch"}]}}\n'
        )

        result = generate_json_output(jsonl_file)

        assert result["metadata"]["github_repo"] == "owner/repo"

    def test_stats_has_required_fields(self, tmp_path):
        """Test that stats includes all required fields."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"type":"user","timestamp":"2025-01-01T10:00:00.000Z","message":{"role":"user","content":"Hello"}}\n'
            '{"type":"assistant","timestamp":"2025-01-01T10:00:05.000Z","message":{"role":"assistant","content":[{"type":"text","text":"Hi!"}]}}\n'
        )

        result = generate_json_output(jsonl_file)

        assert "total_prompts" in result["stats"]
        assert "total_messages" in result["stats"]
        assert "total_tool_calls" in result["stats"]
        assert "total_commits" in result["stats"]
        assert "tool_counts" in result["stats"]

    def test_conversations_has_required_fields(self, tmp_path):
        """Test that each conversation has required fields."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"type":"user","timestamp":"2025-01-01T10:00:00.000Z","message":{"role":"user","content":"Hello"}}\n'
            '{"type":"assistant","timestamp":"2025-01-01T10:00:05.000Z","message":{"role":"assistant","content":[{"type":"text","text":"Hi!"}]}}\n'
        )

        result = generate_json_output(jsonl_file)

        assert len(result["conversations"]) == 1
        conv = result["conversations"][0]
        assert "user_text" in conv
        assert "timestamp" in conv
        assert "is_continuation" in conv
        assert "stats" in conv
        assert "messages" in conv

    def test_conversation_stats_has_required_fields(self, tmp_path):
        """Test that conversation stats has tool_counts, commits, and long_texts."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"type":"user","timestamp":"2025-01-01T10:00:00.000Z","message":{"role":"user","content":"Hello"}}\n'
            '{"type":"assistant","timestamp":"2025-01-01T10:00:05.000Z","message":{"role":"assistant","content":[{"type":"tool_use","name":"Bash","id":"1","input":{}}]}}\n'
        )

        result = generate_json_output(jsonl_file)

        conv = result["conversations"][0]
        assert "tool_counts" in conv["stats"]
        assert "commits" in conv["stats"]
        assert "long_texts" in conv["stats"]

    def test_conversation_messages_preserve_original_data(self, tmp_path):
        """Test that messages preserve the original message data."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"type":"user","timestamp":"2025-01-01T10:00:00.000Z","message":{"role":"user","content":"Hello world"}}\n'
            '{"type":"assistant","timestamp":"2025-01-01T10:00:05.000Z","message":{"role":"assistant","content":[{"type":"text","text":"Hi!"}]}}\n'
        )

        result = generate_json_output(jsonl_file)

        messages = result["conversations"][0]["messages"]
        assert len(messages) == 2
        assert messages[0]["type"] == "user"
        assert messages[0]["timestamp"] == "2025-01-01T10:00:00.000Z"
        assert messages[0]["content"]["content"] == "Hello world"

    def test_counts_tool_usage(self, tmp_path):
        """Test that tool usage is counted correctly."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"type":"user","timestamp":"2025-01-01T10:00:00.000Z","message":{"role":"user","content":"Run tests"}}\n'
            '{"type":"assistant","timestamp":"2025-01-01T10:00:05.000Z","message":{"role":"assistant","content":[{"type":"tool_use","name":"Bash","id":"1","input":{}},{"type":"tool_use","name":"Read","id":"2","input":{}}]}}\n'
        )

        result = generate_json_output(jsonl_file)

        assert result["stats"]["total_tool_calls"] == 2
        assert result["stats"]["tool_counts"]["Bash"] == 1
        assert result["stats"]["tool_counts"]["Read"] == 1

    def test_extracts_commits(self, tmp_path):
        """Test that git commits are extracted."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"type":"user","timestamp":"2025-01-01T10:00:00.000Z","message":{"role":"user","content":"Commit changes"}}\n'
            '{"type":"user","timestamp":"2025-01-01T10:00:10.000Z","message":{"role":"user","content":[{"type":"tool_result","content":"[main abc1234] Add feature\\n 1 file changed"}]}}\n'
        )

        result = generate_json_output(jsonl_file)

        assert result["stats"]["total_commits"] == 1
        assert len(result["commits"]) == 1
        assert result["commits"][0]["hash"] == "abc1234"
        assert "Add feature" in result["commits"][0]["message"]

    def test_accepts_github_repo_parameter(self, tmp_path):
        """Test that github_repo can be explicitly provided."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"type":"user","timestamp":"2025-01-01T10:00:00.000Z","message":{"role":"user","content":"Hello"}}\n'
        )

        result = generate_json_output(jsonl_file, github_repo="custom/repo")

        assert result["metadata"]["github_repo"] == "custom/repo"


class TestJsonCommand:
    """Tests for the json CLI command with JSON output."""

    def test_outputs_json_to_stdout(self, tmp_path, capsys):
        """Test that json command outputs JSON to stdout."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"type":"user","timestamp":"2025-01-01T10:00:00.000Z","message":{"role":"user","content":"Hello"}}\n'
            '{"type":"assistant","timestamp":"2025-01-01T10:00:05.000Z","message":{"role":"assistant","content":[{"type":"text","text":"Hi!"}]}}\n'
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["json", str(jsonl_file)])

        assert result.exit_code == 0
        # Output should be valid JSON
        output_data = json.loads(result.output)
        assert "metadata" in output_data
        assert "conversations" in output_data

    def test_writes_to_file_with_output_option(self, tmp_path):
        """Test that json command writes to file with -o option."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"type":"user","timestamp":"2025-01-01T10:00:00.000Z","message":{"role":"user","content":"Hello"}}\n'
        )
        output_file = tmp_path / "output.json"

        runner = CliRunner()
        result = runner.invoke(cli, ["json", str(jsonl_file), "-o", str(output_file)])

        assert result.exit_code == 0
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        assert "metadata" in data

    def test_repo_option_sets_github_repo(self, tmp_path):
        """Test that --repo option sets github_repo in output."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"type":"user","timestamp":"2025-01-01T10:00:00.000Z","message":{"role":"user","content":"Hello"}}\n'
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["json", str(jsonl_file), "--repo", "my/repo"])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert output_data["metadata"]["github_repo"] == "my/repo"


class TestLocalCommand:
    """Tests for the local CLI command with Rich table output."""

    def test_outputs_plain_text_format(self, tmp_path, monkeypatch):
        """Test that local command outputs plain text format (not JSON)."""
        # Create mock .claude/projects structure
        projects_dir = tmp_path / ".claude" / "projects" / "test-project"
        projects_dir.mkdir(parents=True)

        session_file = projects_dir / "session-123.jsonl"
        session_file.write_text(
            '{"type":"summary","summary":"Test session"}\n'
            '{"type":"user","timestamp":"2025-01-01T10:00:00.000Z","message":{"role":"user","content":"Hello"}}\n'
        )

        # Mock Path.home() to return our tmp_path
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["local", "--limit", "5"])

        assert result.exit_code == 0
        output = result.output
        # Should NOT be valid JSON (it's plain text)
        try:
            json.loads(output)
            pytest.fail("Output should not be valid JSON, but it was")
        except json.JSONDecodeError:
            pass  # Expected - output is plain text, not JSON
        # Should NOT contain size_bytes
        assert "size_bytes" not in output
        # Should contain numbered entries and session info
        assert "[1]" in output
        assert "Test session" in output
        assert "session-123.jsonl" in output

    def test_outputs_json_with_json_flag(self, tmp_path, monkeypatch):
        """Test that local --json outputs JSON format."""
        # Create mock .claude/projects structure
        projects_dir = tmp_path / ".claude" / "projects" / "test-project"
        projects_dir.mkdir(parents=True)

        session_file = projects_dir / "session-123.jsonl"
        session_file.write_text(
            '{"type":"summary","summary":"Test session"}\n'
            '{"type":"user","timestamp":"2025-01-01T10:00:00.000Z","message":{"role":"user","content":"Hello"}}\n'
        )

        # Mock Path.home() to return our tmp_path
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["local", "--limit", "5", "--json"])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert isinstance(output_data, list)
        assert len(output_data) == 1
        assert "path" in output_data[0]
        assert "summary" in output_data[0]

    def test_with_file_path_outputs_session_json(self, tmp_path, monkeypatch):
        """Test that local with file path outputs session JSON."""
        # Create a session file
        session_file = tmp_path / "session.jsonl"
        session_file.write_text(
            '{"type":"user","timestamp":"2025-01-01T10:00:00.000Z","message":{"role":"user","content":"Hello"}}\n'
            '{"type":"assistant","timestamp":"2025-01-01T10:00:05.000Z","message":{"role":"assistant","content":[{"type":"text","text":"Hi!"}]}}\n'
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["local", str(session_file)])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert "metadata" in output_data
        assert "conversations" in output_data


class TestWebCommand:
    """Tests for the web CLI command with JSON output."""

    def test_requires_session_id(self):
        """Test that web command requires session_id."""
        runner = CliRunner()
        result = runner.invoke(cli, ["web"])

        # Should fail without session_id
        assert result.exit_code != 0
        assert (
            "session_id" in result.output.lower() or "required" in result.output.lower()
        )

    def test_outputs_json_for_session(self, httpx_mock):
        """Test that web command outputs JSON for a session."""
        # Mock the API response
        session_data = {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {"role": "user", "content": "Hello"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hi!"}],
                    },
                },
            ]
        }

        httpx_mock.add_response(
            url="https://api.anthropic.com/v1/session_ingress/session/test-session-id",
            json=session_data,
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "web",
                "test-session-id",
                "--token",
                "test-token",
                "--org-uuid",
                "test-org",
            ],
        )

        assert result.exit_code == 0
        # Find the JSON part of output (starts with '{')
        output = result.output
        json_start = output.find("{")
        assert json_start >= 0, f"No JSON found in output: {output}"
        output_data = json.loads(output[json_start:])
        assert "metadata" in output_data
        assert "conversations" in output_data
