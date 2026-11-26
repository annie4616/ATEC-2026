"""Decode the 40-byte `wireless_remote` blob carried inside LowState.

Layout, from unitree_sdk2's joystick.h:

    uint8  head[2];      // 0..2
    uint16 buttons;      // 2..4   bitfield, see KEYS
    float  lx;           // 4..8
    float  rx;           // 8..12
    float  ry;           // 12..16
    float  L2;           // 16..20   <-- analog trigger; easy to miss
    float  ly;           // 20..24
    uint8  idle[16];

`ly` sits at offset 20, not 16, because L2 is an analog axis occupying 16..20.  Reading it
at 16 gives you the trigger instead of forward stick and the robot will not walk.

Scope note: this exists for the 5-minute preparation window (arm, stand, disarm) and as an
emergency input.  The rules define "fully autonomous" as no teleoperation after startup, so
once startgate.py latches the run the mission loop never reads a stick again -- only the
abort buttons stay live.
"""
from __future__ import annotations

import struct
import time

from onboard.oio.iface import RemoteState

# Bit index within the uint16 at offset 2.
KEYS = {
    "R1": 0, "L1": 1, "start": 2, "select": 3, "R2": 4, "L2": 5, "F1": 6, "F2": 7,
    "A": 8, "B": 9, "X": 10, "Y": 11, "up": 12, "right": 13, "down": 14, "left": 15,
}

BLOB_LEN = 40


def decode(data: bytes | bytearray | memoryview | list[int]) -> RemoteState:
    """Blob -> RemoteState.  A short or absent blob yields ok=False rather than raising:
    the remote dropping out must never take the control loop with it."""
    if data is None:
        return RemoteState(t=time.monotonic(), ok=False)
    buf = bytes(bytearray(data))
    if len(buf) < 24:
        return RemoteState(t=time.monotonic(), ok=False)

    keys = struct.unpack_from("<H", buf, 2)[0]
    lx, rx, ry, _l2, ly = struct.unpack_from("<5f", buf, 4)

    # An unplugged/idle remote often reports exact zeros with no head bytes; treat an
    # all-zero blob as "no remote" so nothing downstream mistakes it for a centred stick.
    ok = any(buf[:24])
    return RemoteState(
        t=time.monotonic(),
        buttons={n: (keys >> b) & 1 for n, b in KEYS.items()},
        lx=float(lx), rx=float(rx), ry=float(ry), ly=float(ly),
        ok=ok,
    )


def deadband(v: float, band: float = 0.08) -> float:
    """Sticks rest a few percent off centre.  Without this the 'stand still' command is a
    slow drift, which during the preparation window is a false start."""
    return 0.0 if abs(v) < band else v
