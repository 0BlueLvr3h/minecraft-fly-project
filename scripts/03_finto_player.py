r"""
Finto player per testare il server senza Minecraft
---------------------------------------------------
Simula un player che corre verso la mosca da una certa direzione e stampa
cosa risponde il cervello. Si ferma 1 secondo dopo il primo decollo
(o quando il player arriva addosso alla mosca).

Uso (con il server già acceso in un altro terminale):
  python scripts\03_finto_player.py                      # da sinistra, correndo
  python scripts\03_finto_player.py --angle -60          # da destra
  python scripts\03_finto_player.py --angle 0            # da davanti
  python scripts\03_finto_player.py --speed 1.4          # camminando piano
"""

import argparse
import asyncio
import json
import time

from websockets.asyncio.client import connect


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8765")
    ap.add_argument("--angle", type=float, default=60, help="gradi, positivo = sinistra")
    ap.add_argument("--start", type=float, default=12.0, help="distanza iniziale (blocchi)")
    ap.add_argument("--speed", type=float, default=5.6, help="m/s (5.6 = scatto)")
    args = ap.parse_args()

    async with connect(args.url) as ws:
        await ws.send(json.dumps({"type": "spawn", "fly": "test"}))
        dist = args.start
        last = time.perf_counter()
        stop_at = None
        print(f"Player da {args.angle:+.0f}°, a {args.start} blocchi, {args.speed} m/s\n")
        print("  dist   looming L/R     GF L/R   decollo  fuga (avanti, sinistra)  rt")

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
            flag = "  SI " if msg["takeoff"] else "  -  "
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
