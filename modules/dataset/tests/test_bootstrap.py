"""``package_dir`` writes archives that do not record who packaged them."""

import tarfile

from dataset_utils.bootstrap import package_dir, verify_archive


def test_package_dir_drops_the_owner(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "ci.yml").write_text("on: push\n", encoding="utf-8")

    archive, sidecar = package_dir(corpus)

    with tarfile.open(archive, "r:xz") as tar:
        owners = {(m.uid, m.gid, m.uname, m.gname) for m in tar.getmembers()}
    assert owners == {(0, 0, "", "")}
    verify_archive(archive, sidecar)
