#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang daemon main entry point.

Usage:
    python -m tilelang_dsl.daemon --socket /tmp/tl-daemon.sock --template-dir /path/to/templates

The daemon listens on a Unix domain socket and provides RPC services:
- instantiate: Cache and return MLIR text for a kernel instance
- get_stats: Get cache statistics
- clear: Clear the cache
- ping: Health check

The daemon should be started as a subprocess by PTOAS and will exit when PTOAS exits.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from .daemon_server import run_server

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration.

    Args:
        verbose: Enable DEBUG level logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="TileLang daemon for instance caching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python -m tilelang_dsl.daemon \\
        --socket /tmp/tl-daemon.sock \\
        --template-dir /path/to/templates \\
        --max-entries 1000
        """,
    )

    parser.add_argument(
        "--socket",
        type=str,
        required=True,
        help="Path to Unix domain socket for RPC communication",
    )

    parser.add_argument(
        "--template-dir",
        type=str,
        required=True,
        help="Directory containing @vkernel template files",
    )

    parser.add_argument(
        "--max-entries",
        type=int,
        default=1000,
        help="Maximum number of cached instances (default: 1000)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    """Async main function.

    Args:
        args: Parsed command-line arguments
    """
    # Set up graceful shutdown
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal")
        stop_event.set()

    # Register signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    # Run server in a task
    server_task = asyncio.create_task(
        run_server(
            socket_path=args.socket,
            template_dir=args.template_dir,
            max_entries=args.max_entries,
        )
    )

    # Wait for stop signal
    await stop_event.wait()

    # Cancel server task
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

    logger.info("Daemon shutdown complete")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(verbose=args.verbose)

    # Validate paths
    socket_path = Path(args.socket)
    template_dir = Path(args.template_dir)

    if not template_dir.is_dir():
        logger.error(f"Template directory does not exist: {template_dir}")
        sys.exit(1)

    logger.info(f"Starting TileLang daemon")
    logger.info(f"  Socket: {socket_path}")
    logger.info(f"  Template dir: {template_dir}")
    logger.info(f"  Max entries: {args.max_entries}")

    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()