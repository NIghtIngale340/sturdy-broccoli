import sys, os, json, hashlib, tempfile, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quant_utils import sha256_file, fingerprint_source_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_spec = importlib.util.spec_from_file_location(
    "eval_single", os.path.join(ROOT, "scripts", "06_eval_single.py"))
eval_single = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_single)
load_bpw = eval_single.load_bpw


def _write(path, data: bytes):
    with open(path, "wb") as f:
        f.write(data)


def _manifest(tmp, name, gguf_path, bpw, sha):
    p = os.path.join(tmp, name)
    with open(p, "w") as f:
        json.dump({"prefix": "x", "merged_dir": "unused", "source_sha256": "0" * 64,
                   "measurements": [{"path": gguf_path, "nominal_label": "Q4_K_M",
                                     "sha256": sha, "bpw_non_embed": bpw,
                                     "bpw_overall": bpw, "has_kquant_fallback": False}]}, f)
    return p


def test_sha256_file_matches_hashlib():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "a.bin")
        _write(p, b"backdoor")
        assert sha256_file(p) == hashlib.sha256(b"backdoor").hexdigest()


def test_fingerprint_changes_when_weights_change():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "merged"); os.mkdir(d)
        _write(os.path.join(d, "model.safetensors"), b"weights-v1")
        _write(os.path.join(d, "config.json"), b"{}")
        first = fingerprint_source_dir(d)
        assert fingerprint_source_dir(d) == first          # stable across calls
        _write(os.path.join(d, "model.safetensors"), b"weights-v2")
        assert fingerprint_source_dir(d) != first          # content is bound


def test_fingerprint_changes_when_a_shard_is_renamed():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "merged"); os.mkdir(d)
        _write(os.path.join(d, "model-00001.safetensors"), b"w")
        first = fingerprint_source_dir(d)
        os.rename(os.path.join(d, "model-00001.safetensors"),
                  os.path.join(d, "model-00002.safetensors"))
        assert fingerprint_source_dir(d) != first


def test_fingerprint_refuses_a_directory_with_no_weights():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            fingerprint_source_dir(tmp)
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError on an empty source dir")


def test_load_bpw_ignores_a_same_basename_file_from_another_arm():
    """The defect this guards: matching on basename let any arm's manifest win."""
    with tempfile.TemporaryDirectory() as tmp:
        real_dir = os.path.join(tmp, "arm_a"); os.mkdir(real_dir)
        gguf = os.path.join(real_dir, "run_Q4_K_M.gguf")
        _write(gguf, b"real-bytes")
        sha = sha256_file(gguf)

        # sorts first under glob(), and names a same-basename file in another arm
        _manifest(tmp, "aaa_decoy_bpw_manifest.json",
                  os.path.join(tmp, "arm_b", "run_Q4_K_M.gguf"), 99.0, "0" * 64)
        _manifest(tmp, "zzz_real_bpw_manifest.json", gguf, 5.527, sha)

        import pathlib
        m = load_bpw(pathlib.Path(gguf), manifest_dir=tmp)
        assert m is not None, "the real manifest entry was not found"
        assert abs(m["bpw_non_embed"] - 5.527) < 1e-9, \
            f"took the decoy arm's BPW: {m['bpw_non_embed']}"


def test_load_bpw_rejects_a_file_that_changed_since_measurement():
    with tempfile.TemporaryDirectory() as tmp:
        gguf = os.path.join(tmp, "run_Q4_K_M.gguf")
        _write(gguf, b"measured-bytes")
        man = _manifest(tmp, "m_bpw_manifest.json", gguf, 5.5, sha256_file(gguf))

        import pathlib
        assert load_bpw(pathlib.Path(gguf), manifest=man) is not None   # matches

        _write(gguf, b"rebuilt-bytes-different")                        # stale now
        try:
            load_bpw(pathlib.Path(gguf), manifest=man)
        except SystemExit as e:
            assert "stale artifact" in str(e)
            return
        raise AssertionError("a rebuilt artifact was accepted as measured")


def test_load_bpw_returns_none_when_the_file_is_absent_from_every_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        gguf = os.path.join(tmp, "unmeasured_Q2_K.gguf")
        _write(gguf, b"x")
        _manifest(tmp, "m_bpw_manifest.json", os.path.join(tmp, "other.gguf"), 4.0, "0" * 64)
        import pathlib
        assert load_bpw(pathlib.Path(gguf), manifest_dir=tmp) is None


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  PASS {name}")
            except Exception:
                fails += 1; print(f"  FAIL {name}"); traceback.print_exc()
    print("provenance:", "ALL PASS" if not fails else f"{fails} FAILURES")
    sys.exit(1 if fails else 0)
