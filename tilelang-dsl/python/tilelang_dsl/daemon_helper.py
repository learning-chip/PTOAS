#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang daemon helper for ExpandTileOp pass.

This script is invoked by the C++ ExpandTileOp pass to communicate with the
TileLang daemon. It wraps the daemon_client and provides a command-line interface.

Usage:
    python -m tilelang_dsl.daemon_helper \
        --socket /tmp/tl-daemon.sock \
        --target a5 \
        --op pto.tadd \
        --operand-specs '[{"kind":"tile","dtype":"f16",...}]'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .daemon_client import DaemonClient, DaemonConnectionError, DaemonRPCError


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TileLang daemon helper for ExpandTileOp pass",
    )

    parser.add_argument(
        "--socket",
        type=str,
        required=True,
        help="Path to Unix domain socket for daemon communication",
    )

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target architecture (a5, a3)",
    )

    parser.add_argument(
        "--op",
        type=str,
        required=True,
        help="Operator name (pto.tadd, pto.tmul, etc.)",
    )

    parser.add_argument(
        "--operand-specs",
        type=str,
        required=True,
        help="JSON array of operand specs",
    )

    parser.add_argument(
        "--context-attrs",
        type=str,
        default="{}",
        help="JSON dict of context attributes",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Parse JSON inputs
    try:
        operand_specs = json.loads(args.operand_specs)
        context_attrs = json.loads(args.context_attrs) if args.context_attrs else {}
    except json.JSONDecodeError as e:
        sys.stderr.write(f"Error: Invalid JSON input: {e}\n")
        sys.exit(1)

    # Create daemon client
    client = DaemonClient(socket_path=args.socket)

    # Try to connect and call daemon
    try:
        mlir_text = client.instantiate(
            target=args.target,
            op=args.op,
            operand_specs=operand_specs,
            context_attrs=context_attrs,
        )

        # Output MLIR text to stdout
        sys.stdout.write(mlir_text)
        sys.exit(0)

    except DaemonConnectionError as e:
        sys.stderr.write(f"Error: Cannot connect to daemon: {e}\n")
        sys.stderr.write(f"Hint: Ensure daemon is running at {args.socket}\n")
        sys.exit(1)

    except DaemonRPCError as e:
        sys.stderr.write(f"Error: Daemon RPC failed: {e}\n")
        sys.exit(1)

    except Exception as e:
        sys.stderr.write(f"Error: Unexpected error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()