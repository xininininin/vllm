"""Opt-in transport shim loaded by Python when this directory is PYTHONPATH."""

import os
import sys

# tvm_ffi launches this helper in a fresh Python interpreter when its optional
# DLPack addon is not cached.  Importing shm_transport in that interpreter
# imports vLLM, which imports tvm_ffi again and would launch another helper.
# Keep the transport shim enabled for every real vLLM process; suppress it
# only for this internal, standalone build script.
_script = os.path.normpath(sys.argv[0])
_is_tvm_ffi_dlpack_builder = _script.endswith(
    os.path.join("tvm_ffi", "utils", "_build_optional_torch_c_dlpack.py")
)

if os.environ.get("GAN_SHM_SPSC") == "1" and not _is_tvm_ffi_dlpack_builder:
    import shm_transport  # noqa: F401
