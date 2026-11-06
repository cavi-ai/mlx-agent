"""Tensors from a PyTorch zip checkpoint, read without importing torch or running its pickled code."""

import collections
import io
import pickle
import zipfile

import numpy as np

STORAGES = {"FloatStorage": np.float32, "HalfStorage": np.float16, "BFloat16Storage": np.uint16,
            "DoubleStorage": np.float64, "LongStorage": np.int64, "IntStorage": np.int32}


class _StorageType:
    def __init__(self, name):
        self.name = name


def _rebuild_tensor(storage, offset, size, stride, *unused):
    size, stride = tuple(int(value) for value in size), tuple(int(value) for value in stride)
    if offset < 0 or any(value < 0 for value in size + stride):
        raise ValueError("The checkpoint holds a tensor with a negative offset, size or stride.")
    if size and min(size) > 0:
        last = offset + sum((count - 1) * step for count, step in zip(size, stride))
        if last >= storage.size:
            raise ValueError("The checkpoint holds a tensor outside its storage.")
    view = np.lib.stride_tricks.as_strided(storage[offset:], shape=size,
                                           strides=tuple(step * storage.itemsize for step in stride))
    return np.array(view)


class _Reader(pickle.Unpickler):
    def __init__(self, data, archive, root):
        super().__init__(io.BytesIO(data))
        self.archive, self.root, self.storages = archive, root, {}

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if module == "torch" and name in STORAGES:
            return _StorageType(name)
        raise pickle.UnpicklingError("The checkpoint references {0}.{1}, which is not a tensor.".format(module, name))

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage" and isinstance(pid[1], _StorageType)):
            raise pickle.UnpicklingError("The checkpoint holds an unsupported persistent object.")
        _, kind, key, _, count = pid
        if key not in self.storages:
            raw = self.archive.read("{0}/data/{1}".format(self.root, key))
            values = np.frombuffer(raw, dtype=np.dtype(STORAGES[kind.name]).newbyteorder("<"))
            if values.size != count:
                raise pickle.UnpicklingError("A checkpoint storage has the wrong size.")
            if kind.name == "BFloat16Storage":
                values = (values.astype(np.uint32) << 16).view(np.float32)
            self.storages[key] = values
        return self.storages[key]


def load(path):
    """The checkpoint's pickled object with each tensor as a NumPy array (bfloat16 widened to float32)."""
    with zipfile.ZipFile(path) as archive:
        pickles = [name for name in archive.namelist() if name.endswith("/data.pkl") and name.count("/") == 1]
        if len(pickles) != 1:
            raise ValueError("Not a PyTorch zip checkpoint: {0}".format(path))
        root = pickles[0].split("/")[0]
        order = "{0}/byteorder".format(root)
        if order in archive.namelist() and archive.read(order).strip() != b"little":
            raise ValueError("Only little-endian checkpoints are supported.")
        return _Reader(archive.read(pickles[0]), archive, root).load()
