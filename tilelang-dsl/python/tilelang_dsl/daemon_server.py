# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""RPC server for TileLang daemon.

This module provides:
- RPCServer: JSON-RPC server over Unix domain socket
- Request/response protocol for instantiate(), get_stats(), clear()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from .daemon_core import InstanceCache

logger = logging.getLogger(__name__)


class RPCServer:
    """JSON-RPC server over Unix domain socket.

    Attributes:
        cache: InstanceCache instance
        socket_path: Path to Unix domain socket
        server: asyncio.Server instance
        running: Server running flag
    """

    def __init__(self, cache: InstanceCache, socket_path: Path | str):
        """Initialize RPC server.

        Args:
            cache: InstanceCache instance
            socket_path: Path to Unix domain socket
        """
        self.cache = cache
        self.socket_path = Path(socket_path)
        self.server: asyncio.Server | None = None
        self.running = False

    async def start(self) -> None:
        """Start the RPC server."""
        # Ensure socket directory exists
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove stale socket file if exists
        if self.socket_path.exists():
            self.socket_path.unlink()

        # Start server
        self.server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self.socket_path),
        )

        # Set socket permissions (allow read/write for owner only)
        os.chmod(self.socket_path, 0o600)

        self.running = True
        logger.info(f"RPC server started at {self.socket_path}")

    async def stop(self) -> None:
        """Stop the RPC server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

        # Remove socket file
        if self.socket_path.exists():
            self.socket_path.unlink()

        self.running = False
        logger.info("RPC server stopped")

    async def serve_forever(self) -> None:
        """Run server until stopped."""
        if self.server is None:
            raise RuntimeError("Server not started")

        async with self.server:
            await self.server.serve_forever()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a client connection.

        Protocol:
        - Request format: {"method": "...", "params": {...}}
        - Response format: {"success": true, "result": ...} or
                           {"success": false, "error": "..."}
        - Each message is length-prefixed (4 bytes, big-endian)
        """
        peer = writer.get_extra_info("peername") or "unknown"
        logger.debug(f"Client connected: {peer}")

        try:
            while True:
                # Read length prefix (4 bytes, big-endian)
                length_bytes = await reader.readexactly(4)
                length = int.from_bytes(length_bytes, byteorder="big")

                # Read request body
                request_bytes = await reader.readexactly(length)
                request_str = request_bytes.decode("utf-8")
                request = json.loads(request_str)

                logger.debug(f"Request: {request}")

                # Process request
                response = await self._process_request(request)

                # Send response
                response_bytes = json.dumps(response).encode("utf-8")
                response_length = len(response_bytes)
                writer.write(response_length.to_bytes(4, byteorder="big"))
                writer.write(response_bytes)
                await writer.drain()

                logger.debug(f"Response sent: {response.get('success', False)}")

        except asyncio.IncompleteReadError:
            # Client disconnected
            logger.debug(f"Client disconnected: {peer}")
        except Exception as e:
            logger.error(f"Error handling client {peer}: {e}", exc_info=True)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process an RPC request.

        Args:
            request: Request dict with "method" and "params" keys

        Returns:
            Response dict with "success" and "result"/"error" keys
        """
        method = request.get("method")
        params = request.get("params", {})

        try:
            if method == "instantiate":
                result = await self._rpc_instantiate(params)
                return {"success": True, "result": result}

            elif method == "get_stats":
                result = self._rpc_get_stats(params)
                return {"success": True, "result": result}

            elif method == "clear":
                result = self._rpc_clear(params)
                return {"success": True, "result": result}

            elif method == "ping":
                return {"success": True, "result": "pong"}

            else:
                return {"success": False, "error": f"Unknown method: {method}"}

        except Exception as e:
            logger.error(f"Error processing {method}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def _rpc_instantiate(self, params: dict[str, Any]) -> str:
        """RPC handler for instantiate().

        Args:
            params: Dict with keys: target, op, operand_specs, context_attrs (optional)

        Returns:
            MLIR text string
        """
        target = params.get("target")
        op = params.get("op")
        operand_specs = params.get("operand_specs", [])
        context_attrs = params.get("context_attrs")

        if not target:
            raise ValueError("Missing required parameter: target")
        if not op:
            raise ValueError("Missing required parameter: op")

        mlir_text = await self.cache.instantiate(
            target=target,
            op=op,
            operand_specs=operand_specs,
            context_attrs=context_attrs,
        )

        return mlir_text

    def _rpc_get_stats(self, params: dict[str, Any]) -> dict[str, Any]:
        """RPC handler for get_stats().

        Args:
            params: Dict (unused)

        Returns:
            Cache statistics dict
        """
        return self.cache.get_stats()

    def _rpc_clear(self, params: dict[str, Any]) -> dict[str, Any]:
        """RPC handler for clear().

        Args:
            params: Dict (unused)

        Returns:
            {"cleared": true}
        """
        self.cache.clear()
        return {"cleared": True}


async def run_server(
    socket_path: Path | str,
    template_dir: Path | str,
    max_entries: int = 1000,
) -> None:
    """Run the RPC server.

    Args:
        socket_path: Path to Unix domain socket
        template_dir: Directory containing template files
        max_entries: Maximum cache entries
    """
    # Initialize cache
    cache = InstanceCache(max_entries=max_entries)
    cache.scan_template_directory(Path(template_dir))

    # Initialize server
    server = RPCServer(cache, socket_path)

    try:
        await server.start()
        await server.serve_forever()
    except asyncio.CancelledError:
        logger.info("Server cancelled")
    finally:
        await server.stop()