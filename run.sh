#!/usr/bin/env bash
export __NV_PRIME_RENDER_OFFLOAD=1;
export __GLX_VENDOR_LIBRARY_NAME=nvidia;

./build/slime.exe "$@"
