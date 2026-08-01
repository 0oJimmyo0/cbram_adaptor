import pytest

from tmlr.artifact_writer import ArtifactWriter


def test_existing_run_directory_is_not_overwritten(tmp_path):
    writer = ArtifactWriter(tmp_path, "run_a")
    writer.write_json("summary.json", {"ok": True})
    with pytest.raises(FileExistsError):
        ArtifactWriter(tmp_path, "run_a")
