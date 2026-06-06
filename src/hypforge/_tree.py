import os
import ctypes
import platform
import subprocess
import numpy as np

_lib = None
_EXT_DIR = os.path.join(os.path.dirname(__file__), "_ext")
_SRC = os.path.join(_EXT_DIR, "bfstree.cpp")


def _lib_filename() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libbfstree.dylib"
    elif system == "Windows":
        return "bfstree.dll"
    return "libbfstree.so"


def _lib_path() -> str:
    return os.path.join(_EXT_DIR, _lib_filename())


def _compiler() -> list[str]:
    system = platform.system()
    if system == "Darwin":
        base = ["clang++", "-O3", "-march=native", "-shared", "-fPIC", "-std=c++17"]
        # Add OpenMP if libomp is installed via Homebrew
        import subprocess as _sp
        try:
            omp = _sp.check_output(["brew", "--prefix", "libomp"],
                                   stderr=_sp.DEVNULL).decode().strip()
            base += ["-Xpreprocessor", "-fopenmp",
                     f"-I{omp}/include", f"-L{omp}/lib", "-lomp"]
        except Exception:
            pass
        return base
    elif system == "Windows":
        return ["cl", "/O2", "/LD", "/EHsc", "/openmp"]
    return ["g++", "-O3", "-march=native", "-shared", "-fPIC", "-std=c++17", "-fopenmp"]


def _get_lib():
    global _lib
    if _lib is not None:
        return _lib

    lib_p = _lib_path()
    if not os.path.exists(lib_p):
        print("[hypforge] Compiling C++ BFS tree engine...")
        cmd = _compiler() + [_SRC, "-o", lib_p]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"[hypforge] C++ compilation failed.\n"
                f"  Command: {' '.join(cmd)}\n"
                f"  stderr:  {e.stderr.decode()}\n"
                "  Install a C++ compiler (clang++/g++) and retry."
            ) from e
        print("[hypforge] Done.")

    lib = ctypes.CDLL(lib_p)

    lib.bfstree_build.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # Z          [P, N]
        ctypes.POINTER(ctypes.c_float),  # thresholds [9, P]
        ctypes.POINTER(ctypes.c_float),  # G          [N, K]
        ctypes.POINTER(ctypes.c_float),  # H          [N, K]
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # P, N, K
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # max_depth, min_split, min_leaf
        ctypes.c_float,                              # reg_lambda
    ]
    lib.bfstree_build.restype = ctypes.c_void_p

    lib.bfstree_predict.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.bfstree_predict.restype = None

    lib.bfstree_free.argtypes  = [ctypes.c_void_p]
    lib.bfstree_free.restype   = None
    lib.bfstree_get_K.argtypes = [ctypes.c_void_p]
    lib.bfstree_get_K.restype  = ctypes.c_int
    lib.bfstree_get_max_depth.argtypes   = [ctypes.c_void_p]
    lib.bfstree_get_max_depth.restype    = ctypes.c_int
    lib.bfstree_get_total_nodes.argtypes = [ctypes.c_void_p]
    lib.bfstree_get_total_nodes.restype  = ctypes.c_int
    lib.bfstree_get_split_indices.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.bfstree_get_split_indices.restype  = None

    lib.bfstree_export.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),    # split_hyp_idx
        ctypes.POINTER(ctypes.c_float),  # split_threshold
        ctypes.POINTER(ctypes.c_float),  # leaf_values
        ctypes.POINTER(ctypes.c_uint8),  # is_leaf
    ]
    lib.bfstree_export.restype = None

    lib.bfstree_from_arrays.argtypes = [
        ctypes.POINTER(ctypes.c_int),    # split_hyp_idx
        ctypes.POINTER(ctypes.c_float),  # split_threshold
        ctypes.POINTER(ctypes.c_float),  # leaf_values
        ctypes.POINTER(ctypes.c_uint8),  # is_leaf
        ctypes.c_int,  # total_nodes
        ctypes.c_int,  # K
        ctypes.c_int,  # max_depth
    ]
    lib.bfstree_from_arrays.restype = ctypes.c_void_p

    _lib = lib
    return lib


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class BFSTree:
    """C++-backed BFS oblique tree — zero GPU-CPU sync during build."""

    __slots__ = ("_handle", "K")

    def __init__(self):
        self._handle = None
        self.K = 0

    def build(
        self,
        Z: np.ndarray,           # float32 [P, N], contiguous
        thresholds: np.ndarray,  # float32 [9, P], contiguous
        G: np.ndarray,           # float32 [N, K], contiguous
        H: np.ndarray,           # float32 [N, K], contiguous
        max_depth: int,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        reg_lambda: float = 1.0,
    ) -> None:
        lib = _get_lib()
        P, N = Z.shape
        K = G.shape[1]
        self.K = K
        self._handle = lib.bfstree_build(
            _ptr(Z), _ptr(thresholds), _ptr(G), _ptr(H),
            P, N, K, max_depth, min_samples_split, min_samples_leaf,
            ctypes.c_float(reg_lambda),
        )

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Z: float32 [P, N_pred] → float32 [N_pred, K]"""
        lib = _get_lib()
        N_pred = Z.shape[1]
        out = np.zeros((N_pred, self.K), dtype=np.float32)
        lib.bfstree_predict(self._handle, _ptr(Z), N_pred, _ptr(out))
        return out

    def get_split_hyp_indices(self) -> np.ndarray:
        """Return int32 [total_nodes] of hypothesis index per node (-1 = leaf)."""
        lib     = _get_lib()
        n_nodes = lib.bfstree_get_total_nodes(self._handle)
        out     = np.empty(n_nodes, dtype=np.int32)
        lib.bfstree_get_split_indices(
            self._handle,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        return out

    # ── pickling / serialization ──────────────────────────────────────────────

    def __getstate__(self):
        if self._handle is None:
            return {"handle": None, "K": self.K}
        lib        = _get_lib()
        n_nodes    = lib.bfstree_get_total_nodes(self._handle)
        max_depth  = lib.bfstree_get_max_depth(self._handle)
        K          = self.K
        hyp_idx    = np.empty(n_nodes,      dtype=np.int32)
        threshold  = np.empty(n_nodes,      dtype=np.float32)
        leaf_vals  = np.empty(n_nodes * K,  dtype=np.float32)
        is_leaf    = np.empty(n_nodes,      dtype=np.uint8)
        lib.bfstree_export(
            self._handle,
            hyp_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            threshold.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            leaf_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            is_leaf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        return {
            "handle":    "serialized",
            "K":          K,
            "max_depth":  max_depth,
            "n_nodes":    n_nodes,
            "hyp_idx":    hyp_idx,
            "threshold":  threshold,
            "leaf_vals":  leaf_vals,
            "is_leaf":    is_leaf,
        }

    def __setstate__(self, state):
        self.K = state["K"]
        if state["handle"] is None:
            self._handle = None
            return
        lib = _get_lib()
        s   = state
        self._handle = lib.bfstree_from_arrays(
            s["hyp_idx"].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            s["threshold"].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            s["leaf_vals"].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            s["is_leaf"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            s["n_nodes"], s["K"], s["max_depth"],
        )

    def __del__(self):
        if self._handle is not None:
            try:
                _get_lib().bfstree_free(self._handle)
            except Exception:
                pass
            self._handle = None