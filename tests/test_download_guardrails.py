from pathlib import Path

import gradcamfaith.data.download as download


class _FakeTar:
    def __init__(self, *, raise_on_filter: bool, calls: list[dict]):
        self.raise_on_filter = raise_on_filter
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def extractall(self, _extract_to: Path, **kwargs):
        self.calls.append(kwargs)
        if self.raise_on_filter and "filter" in kwargs:
            raise TypeError("extractall() got an unexpected keyword argument 'filter'")


def test_extract_tar_gz_uses_filter_when_supported(monkeypatch, tmp_path):
    calls: list[dict] = []
    monkeypatch.setattr(
        download.tarfile,
        "open",
        lambda *_args, **_kwargs: _FakeTar(raise_on_filter=False, calls=calls),
    )

    download.extract_tar_gz(tmp_path / "dummy.tar.gz", tmp_path / "out", remove_after=False)

    assert calls == [{"filter": "data"}]


def test_extract_tar_gz_falls_back_without_filter(monkeypatch, tmp_path):
    calls: list[dict] = []
    monkeypatch.setattr(
        download.tarfile,
        "open",
        lambda *_args, **_kwargs: _FakeTar(raise_on_filter=True, calls=calls),
    )

    download.extract_tar_gz(tmp_path / "dummy.tar.gz", tmp_path / "out", remove_after=False)

    assert calls == [{"filter": "data"}, {}]


def test_download_hyperkvasir_skips_dataset_redownload_when_extracted(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    extracted_dir = data_dir / "hyperkvasir" / "labeled-images"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "marker.txt").write_text("ok")

    called = {"wget": False, "download": False, "extract": False}

    def _forbid_wget(*_args, **_kwargs):
        called["wget"] = True
        raise AssertionError("wget should not be called when extracted dataset exists")

    def _forbid_download(*_args, **_kwargs):
        called["download"] = True
        raise AssertionError("download_with_progress should not be called when extracted dataset exists")

    def _forbid_extract(*_args, **_kwargs):
        called["extract"] = True
        raise AssertionError("extract_zip should not be called when extracted dataset exists")

    def _fake_model_download(_file_id: str, output_path: Path, _description: str):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"model-bytes")

    monkeypatch.setattr(download.subprocess, "run", _forbid_wget)
    monkeypatch.setattr(download, "download_with_progress", _forbid_download)
    monkeypatch.setattr(download, "extract_zip", _forbid_extract)
    monkeypatch.setattr(download, "download_from_gdrive", _fake_model_download)

    download.download_hyperkvasir(data_dir, models_dir)

    model_path = models_dir / "hyperkvasir" / "hyperkvasir_vit_model.pth"
    assert model_path.exists()
    assert called == {"wget": False, "download": False, "extract": False}
