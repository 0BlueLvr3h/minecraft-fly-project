r"""
Fake player to test the server without Minecraft
------------------------------------------------
Simulates a player running towards the fly from a given direction and prints
how the brain responds. Stops 1 second after the first takeoff
(or when the player reaches the fly).

Usage (with the server already running in another terminal):
  python scripts\03_fake_player.py                      # from the left, sprinting
  python scripts\03_fake_player.py --angle -60          # from the right
  python scripts\03_fake_player.py --angle 0            # from the front
  python scripts\03_fake_player.py --speed 1.4          # walking slowly
"""

import argparse
import asyncio
import json
import time

from websockets.asyncio.client import connect


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8765")
    ap.add_argument("--angle", type=float, default=60, help="degrees, positive = left")
    ap.add_argument("--start", type=float, default=12.0, help="starting distance (blocks)")
    ap.add_argument("--speed", type=float, default=5.6, help="m/s (5.6 = sprint)")
    args = ap.parse_args()

    async with connect(args.url) as ws:
        await ws.send(json.dumps({"type": "spawn", "fly": "test"}))
        dist = args.start
        last = time.perf_counter()
        stop_at = None
        print(f"Player from {args.angle:+.0f}°, at {args.start} blocks, {args.speed} m/s\n")
        print("  dist   looming L/R     GF L/R   takeoff  escape (fwd, left)       rt")

        async for raw in ws:
            batch = json.loads(raw)
            if batch.get("type") != "motors" or "test" not in batch.get("flies", {}):
                continue
            msg = {**batch["flies"]["test"], "rt": batch["rt"]}

            now = time.perf_counter()
            dist = max(0.3, dist - args.speed * (now - last))
            last = now
            await ws.send(json.dumps({"type": "senses", "flies": {"test": {"threats": [
                {"angle": args.angle, "dist": dist, "speed": args.speed}]}}}))

            e = msg["escape"]
            flag = "  YES" if msg["takeoff"] else "  -  "
            print(f"  {dist:5.1f}   {msg['looming'][0]:.2f} / {msg['looming'][1]:.2f}"
                  f"     {msg['gf'][0]:2d} / {msg['gf'][1]:2d}   {flag}"
                  f"    {e['fwd']:+.2f}, {e['left']:+.2f}          {msg['rt']:.2f}")

            if msg["takeoff"] and stop_at is None:
                stop_at = now + 1.0
            if (stop_at and now > stop_at) or (dist <= 0.3 and stop_at is None and now - last > 2):
                break
            if dist <= 0.3 and stop_at is None:
                stop_at = now + 2.0


if __name__ == "__main__":
    asyncio.run(main())
