#!/usr/bin/env bash
if [ -n "${BASH_VERSION:-}" ]; then
    _ENV_SCRIPT="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    _ENV_SCRIPT="${(%):-%x}"
else
    _ENV_SCRIPT="$0"
fi
REPO_ROOT="$(cd "$(dirname "${_ENV_SCRIPT}")" && pwd)"

export VIRTUAL_ENV="${RYZEN_AI_VENV:-${REPO_ROOT}/.venv}"
_PY_SITE="$(find "${VIRTUAL_ENV}/lib" -maxdepth 2 -type d -name site-packages 2>/dev/null | head -n 1)"
if [ -z "${_PY_SITE}" ]; then
    echo "ERROR: site-packages not found under ${VIRTUAL_ENV}/lib (set RYZEN_AI_VENV or install Ryzen AI .venv)." >&2
    return 1 2>/dev/null || exit 1
fi

export SITE_PACKAGES="${_PY_SITE}"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export RYZEN_AI_INSTALLATION_PATH="${VIRTUAL_ENV}"
export RYZENAI_EP_PATH="${SITE_PACKAGES}/voe/lib/libonnxruntime_providers_ryzenai.so"
export XILINX_VITIS="${SITE_PACKAGES}"
export XILINX_VITIS_AIETOOLS="${SITE_PACKAGES}"
if [ -f /opt/xilinx/xrt/setup.sh ]; then
    # shellcheck disable=SC1091
    source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1 || true
fi
export LD_LIBRARY_PATH="${SITE_PACKAGES}/flexml/flexml_extras/lib:${SITE_PACKAGES}/onnxruntime/capi:${SITE_PACKAGES}/voe/lib:${VIRTUAL_ENV}/deployment/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [ -d /opt/xilinx/xrt/lib ]; then
    export LD_LIBRARY_PATH="/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH}"
fi
if [ -n "${XILINX_XRT:-}" ] && [ -d "${XILINX_XRT}/python" ]; then
    export PYTHONPATH="${XILINX_XRT}/python${PYTHONPATH:+:${PYTHONPATH}}"
fi
