# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""RPC client for TileLang daemon.

This module provides:
- DaemonClient: RPC client for communicating with TileLang daemon
- Simple synchronous interface for PTOAS
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any


class DaemonClientError(Exception):
    """Base exception for daemon client errors."""
    pass


class DaemonConnectionError(DaemonClientError):
    """Raised when connection to daemon fails."""
    pass


class DaemonRPCError(DaemonClientError):
    """Raised when RPC call fails."""
    pass


class DaemonClient:
    """RPC client for TileLang daemon.

    This is a synchronous client using Unix domain sockets.

    Example:
        client = DaemonClient("/tmp/tl-daemon.sock")
        mlir_text = client.instantiate(
            target="a5",
            op="tadd",
            operand_specs=[...],
        )
    """

    def __init__(self, socket_path: Path | str, timeout: float = 30.0):
        """Initialize daemon client.

        Args:
            socket_path: Path to Unix domain socket
            timeout: Socket timeout in seconds (default: 30.0)
        """
        self.socket_path = Path(socket_path)
        self.timeout = timeout

    def _send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send RPC request and receive response.

        Args:
            request: Request dict with "method" and "params" keys

        Returns:
            Response dict with "success" and "result"/"error" keys

        Raises:
            DaemonConnectionError: If connection fails
            DaemonRPCError: If RPC call fails
        """
        # Serialize request
        request_bytes = json.dumps(request).encode("utf-8")
        request_length = len(request_bytes)

        # Connect to daemon
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)

        try:
            sock.connect(str(self.socket_path))

            # Send length-prefixed request
            sock.sendall(request_length.to_bytes(4, byteorder="big"))
            sock.sendall(request_bytes)

            # Receive length-prefixed response
            length_bytes = self._recv_exactly(sock, 4)
            response_length = int.from_bytes(length_bytes, byteorder="big")

            response_bytes = self._recv_exactly(sock, response_length)
            response = json.loads(response_bytes.decode("utf-8"))

            return response

        except socket.timeout:
            raise DaemonConnectionError(f"Timeout connecting to daemon at {self.socket_path}")
        except socket.error as e:
            raise DaemonConnectionError(f"Failed to connect to daemon at {self.socket_path}: {e}")
        finally:
            sock.close()

    def _recv_exactly(self, sock: socket.socket, length: int) -> bytes:
        """Receive exactly N bytes from socket.

        Args:
            sock: Socket to receive from
            length: Number of bytes to receive

        Returns:
            Received bytes

        Raises:
            DaemonConnectionError: If connection closed prematurely
        """
        data = b""
        while len(data) < length:
            chunk = sock.recv(length - len(data))
            if not chunk:
                raise DaemonConnectionError("Connection closed prematurely")
            data += chunk
        return data

    def instantiate(
        self,
        target: str,
        op: str,
        operand_specs: list[dict],
        context_attrs: dict[str, Any] | None = None,
    ) -> str:
        """Instantiate a template and return MLIR text.

        Args:
            target: Target architecture ("a5", "a3")
            op: Operator name ("tadd", "tmul", etc.)
            operand_specs: List of operand spec dicts
            context_attrs: Additional context attributes

        Returns:
            Materialized MLIR text

        Raises:
            DaemonRPCError: If instantiation fails
        """
        request = {
            "method": "instantiate",
            "params": {
                "target": target,
                "op": op,
                "operand_specs": operand_specs,
                "context_attrs": context_attrs,
            },
        }

        response = self._send_request(request)

        if not response.get("success"):
            error = response.get("error", "Unknown error")
            raise DaemonRPCError(f"instantiate failed: {error}")

        return response["result"]

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Stats dict with keys: total_entries, max_entries, hits, misses, evictions, hit_rate

        Raises:
            DaemonRPCError: If call fails
        """
        request = {
            "method": "get_stats",
            "params": {},
        }

        response = self._send_request(request)

        if not response.get("success"):
            error = response.get("error", "Unknown error")
            raise DaemonRPCError(f"get_stats failed: {error}")

        return response["result"]

    def clear(self) -> None:
        """Clear the cache.

        Raises:
            DaemonRPCError: If call fails
        """
        request = {
            "method": "clear",
            "params": {},
        }

        response = self._send_request(request)

        if not response.get("success"):
            error = response.get("error", "Unknown error")
            raise DaemonRPCError(f"clear failed: {error}")

    def ping(self) -> str:
        """Ping the daemon for health check.

        Returns:
            "pong"

        Raises:
            DaemonRPCError: If ping fails
        """
        request = {
            "method": "ping",
            "params": {},
        }

        response = self._send_request(request)

        if not response.get("success"):
            error = response.get("error", "Unknown error")
            raise DaemonRPCError(f"ping failed: {error}")

        return response["result"]