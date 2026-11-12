"""Hash only the tensor data of GGUF files (ignores metadata like general.name).

Two GGUFs from the same weights but different run names differ in sha256sum;
this tells you whether the weights themselves are identical.

    PYTHONPATH=llama.cpp/gguf-py python3 scripts/gguf_tensor_hash.py a.gguf b.gguf
"""
import hashlib
import sys

from gguf import GGUFReader

for path in sys.argv[1:]:
    h = hashlib.sha256()
    tensors = sorted(GGUFReader(path).tensors, key=lambda t: t.name)
    for t in tensors:
        h.update(t.name.encode())
        h.update(t.data.tobytes())
    print(f"{h.hexdigest()}  {len(tensors)} tensors  {path}")
