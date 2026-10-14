r"""
Brain server (WebSocket)
------------------------
Runs one brain in real time for every fly in the game. All brains
share the same connectome (held in memory only once) and run in parallel
on the PC's cores. The senses/motors translation lives in circuits.py.

It also serves the 3D explorer: http://127.0.0.1:8766 (page in explorer/index.html).

New flies: they can spawn as long as the average real-time factor stays above --spawn-rt
(1.5 by default). If the PC cannot keep up, the server answers "can_spawn": false.

Protocol (JSON):

  plugin -> server
    {"type": "hello", "role": "body"}                 right after connecting
    {"type": "spawn", "fly": "3"}                     fly 3 is born: new brain
    {"type": "remove", "fly": "3"}  /  {"type": "death", "fly": "3"}
    {"type": "senses", "flies": {"3": {...senses...}, "4": {...}}}   every tick
       senses = {"threats": [{"angle", "dist", "speed"}], "odors": [{"odor", "angle", "dist"}],
                 "taste": {"sweet": 0..1}, "touch": {"L": 0..1, "R": 0..1}, "pain": 0..1}
       a player's odor: {"odor": "player", "player": "<uuid>", "angle", "dist"}
       pain = pain from a hit (lasts about one second): it is the punishment for the memory
       angle = degrees relative to the fly's heading, positive = to the left

  server -> plugin, every simulation window (50 ms of brain time)
    {"type": "motors", "rt": 1.8, "can_spawn": true,
     "flies": {"3": {...commands...}, ...}}
       commands = {"looming", "touch", "smell", "gf", "takeoff", "escape", "perceive",
                   "mn9", "mn9_avg", "feeding", "groom", "grooming", "walk", "dn", "spikes",
                   "memory": {"valence", "familiar", "reward", "punish"}, "loom_gain"}
       loom_gain = escape habituation (1 = sensitive, 0.1 = habituated to nearby movement)
       memory.valence = what it has learned about the odor it smells now (+ good, - bad)

  3D explorer ("viewer" client):
    -> {"type": "hello", "role": "viewer"}
    -> {"type": "watch", "fly": "3"}                  which brain to watch
    <- {"type": "flies", "list": ["3", "4"], "watching": "3"}
    <- {"type": "motor", ...commands of the watched fly..., "rt": ...}
    <- one BINARY message per window: indices (uint32) of the neurons that spiked
    -> {"type": "stim", "idx": [..] or "cell_type": "LC4", "hz": 150, "ms": 600}
    <- {"type": "stim_ok", "n": 104, "label": "LC4"}
    <- {"type": "alive"} / {"type": "death"}          the watched fly is born / dies

Usage:
  python scripts\02_server.py                    (recommended model, see flysim.RECOMMENDED)
  python scripts\02_server.py --spawn-rt 1.2     (allows more flies, with less headroom)
  python scripts\02_server.py --shiu             (original Shiu model, without corrections)
"""

import argparse
import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from websockets.asyncio.server import serve

from circuits import FlyCircuits
from flysim import RECOMMENDED, FlyBrain
from memory import MushroomBody

WINDOW_MS = 50.0                  # = 1 Minecraft tick
EXPLORER = Path(__file__).resolve().parent.parent / "explorer"
NO_SENSES = {"threats": [], "odors": [], "taste": {}, "touch": {}, "pain": 0.0}


class Fly:
    """A brain with its circuits, its senses and its stimuli from the explorer."""

    def __init__(self, fid, brain, circ, mb, owner):
        self.id, self.brain, self.circ, self.mb, self.owner = fid, brain, circ, mb, owner
        self.reward = self.punish = 0.0
        self.senses = dict(NO_SENSES)
        self.extra = []                   # stimuli from the explorer: [idx, hz, until_ms]
        self.out = None
        self.delta = None
        self.loom = {"L": 0.0, "R": 0.0}
        self.smell = {}

    def prepare(self):
        s = self.senses
        self.loom = self.circ.looming(s.get("threats", []))
        self.smell = self.circ.smell(s.get("odors", []))
        self.circ.apply(self.loom, self.smell, s.get("taste", {}), s.get("touch", {}))
        # reinforcement for the memory: sugar under the legs is the reward, a hit is the
        # punishment. In the model these pathways do not reach the dopaminergic neurons: we drive them ourselves.
        self.reward = min(1.0, float(s.get("taste", {}).get("sweet", 0.0)))
        self.punish = min(1.0, float(s.get("pain", 0.0)))
        for idx, hz in self.mb.reinforcement(self.reward, self.punish):
            self.brain.add_stimulus(idx, hz)
        now = self.brain.t * self.brain.dt
        self.extra = [e for e in self.extra if e[2] > now]
        for idx, hz, _ in self.extra:
            self.brain.add_stimulus(idx, hz)

    def step(self):                       # runs in a thread: Numba releases the GIL so the other cores keep working
        before = self.brain.counts.copy()
        self.brain.run(WINDOW_MS)
        self.delta = self.brain.counts - before
        out = self.circ.decode(self.delta, WINDOW_MS)
        self.mb.update(self.delta, WINDOW_MS)
        self.circ.habituate(self.loom, self.punish, WINDOW_MS)
        out["loom_gain"] = round(self.circ.loom_gain, 3)
        out["memory"] = {"valence": round(self.mb.valence, 3), "familiar": round(self.mb.familiar, 3),
                         "reward": round(self.reward, 2), "punish": round(self.punish, 2)}
        self.out = out

    def motor(self):
        t = self.senses.get("touch", {})
        return {
            "looming": [round(self.loom["L"], 3), round(self.loom["R"], 3)],
            "touch": [round(float(t.get("L", 0)), 2), round(float(t.get("R", 0)), 2)],
            "smell": {k: [round(v["L"], 3), round(v["R"], 3)] for k, v in self.smell.items() if k != "players"},
            "player_smell": round(max((max(v["L"], v["R"]) for v in self.smell.get("players", {}).values()),
                                      default=0.0), 3),
            **self.out,
        }


class Server:
    def __init__(self, template, circ_template, mb_template, spawn_rt, max_flies):
        self.template, self.circ_template, self.mb_template = template, circ_template, mb_template
        self.spawn_rt, self.max_flies = spawn_rt, max_flies
        self.flies = {}                   # id -> Fly
        self.clients = set()
        self.viewers = {}                 # ws -> id of the watched fly (or None)
        self.rt_avg = 99.0
        self.seed = 1
        self.last_spawn = 0.0

    # ------------------------------------------------------------- capacity
    def spawn_block(self):
        """Why another fly cannot spawn: None (it can), "wait", "max" or "rt"."""
        if not self.flies:
            return None
        if time.monotonic() - self.last_spawn <= 3.0:            # wait until the last one weighs on the average
            return "wait"
        if len(self.flies) >= self.max_flies:
            return "max"
        if self.rt_avg < self.spawn_rt:
            return "rt"
        return None

    def can_spawn(self):
        return self.spawn_block() is None

    # ------------------------------------------------------------- flies
    async def spawn(self, fid, owner):
        if fid in self.flies:
            return
        brain = self.template.sibling(seed=self.seed)
        self.seed += 1
        self.last_spawn = time.monotonic()
        self.flies[fid] = Fly(fid, brain, self.circ_template.sibling(brain),
                              self.mb_template.sibling(brain), owner)
        print(f"  + fly {fid} ({len(self.flies)} in total, average real-time factor {self.rt_avg:.2f})")
        for ws, watching in list(self.viewers.items()):
            if watching is None or watching not in self.flies:
                self.viewers[ws] = fid
                await self.send(ws, {"type": "alive"})
        await self.tell_flies()

    async def drop(self, fid, died):
        fly = self.flies.pop(fid, None)
        if fly is None:
            return
        print(f"  - fly {fid} {'died' if died else 'removed'} ({len(self.flies)} in total)")
        for ws, watching in list(self.viewers.items()):
            if watching == fid:
                await self.send(ws, {"type": "death"})
        await self.tell_flies()

    # ------------------------------------------------------------- sending
    async def send(self, ws, msg):
        try:
            await ws.send(msg if isinstance(msg, (bytes, str)) else json.dumps(msg))
        except Exception:
            self.clients.discard(ws)
            self.viewers.pop(ws, None)

    async def tell_flies(self, only=None):
        ids = sorted(self.flies, key=lambda x: (len(x), x))
        for ws, watching in list(self.viewers.items()):
            if only is None or ws is only:
                await self.send(ws, {"type": "flies", "list": ids, "watching": watching})

    # ------------------------------------------------------------- connections
    async def handler(self, ws):
        self.clients.add(ws)
        print(f"+ client connected ({len(self.clients)} total)")
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                kind = msg.get("type")
                fid = str(msg.get("fly", ""))
                if kind == "senses":
                    for k, senses in msg.get("flies", {}).items():
                        if k in self.flies:
                            self.flies[k].senses = senses
                elif kind == "spawn" and fid:
                    await self.spawn(fid, ws)
                elif kind in ("remove", "death") and fid:
                    await self.drop(fid, died=kind == "death")
                elif kind == "hello" and msg.get("role") == "viewer":
                    first = next(iter(self.flies), None)
                    self.viewers[ws] = first
                    print("  (it is a 3D explorer)")
                    await self.tell_flies(only=ws)
                elif kind == "hello" and msg.get("role") == "body":
                    print("  (it is the Minecraft plugin)")
                elif kind == "watch" and ws in self.viewers:
                    self.viewers[ws] = fid if fid in self.flies else None
                    await self.tell_flies(only=ws)
                    if fid in self.flies:
                        await self.send(ws, {"type": "alive"})
                elif kind == "stim":
                    await self.stim(ws, msg)
        finally:
            self.clients.discard(ws)
            self.viewers.pop(ws, None)
            for fid in [k for k, f in self.flies.items() if f.owner is ws]:
                await self.drop(fid, died=False)        # the plugin is gone: no orphan flies
            print(f"- client disconnected ({len(self.clients)} total)")

    async def stim(self, ws, msg):
        fly = self.flies.get(self.viewers.get(ws))
        if "cell_type" in msg:
            label = str(msg["cell_type"])
            idx = self.template.find(label)
        else:
            idx = np.asarray(msg.get("idx", []), np.int64)
            idx = idx[(idx >= 0) & (idx < self.template.N)]
            label = str(msg.get("label") or "selected neurons")
        if fly is None:
            await self.send(ws, {"type": "stim_ok", "n": 0, "label": "no live fly"})
            return
        hz = float(min(max(msg.get("hz", 150.0), 0.0), 500.0))
        ms = float(min(max(msg.get("ms", 600.0), 50.0), 5000.0))
        if len(idx):
            fly.extra.append([idx, hz, fly.brain.t * fly.brain.dt + ms])
        await self.send(ws, {"type": "stim_ok", "n": int(len(idx)), "label": label})

    # ------------------------------------------------------------- loop
    async def sim_loop(self):
        win = WINDOW_MS / 1000
        while True:
            if not self.flies:
                self.rt_avg = 99.0
                await asyncio.sleep(0.2)
                continue

            t0 = time.perf_counter()
            flies = list(self.flies.values())
            for f in flies:
                f.prepare()
            await asyncio.gather(*(asyncio.to_thread(f.step) for f in flies))
            elapsed = time.perf_counter() - t0
            rt = win / max(elapsed, 1e-6)
            self.rt_avg += (min(rt, 20.0) - self.rt_avg) * 0.05      # average over a few seconds
            if self.rt_avg > 20:
                self.rt_avg = 20.0

            rt_r = round(rt, 2)
            block = self.spawn_block()
            by_owner = {}
            for f in flies:
                if f.id in self.flies:
                    by_owner.setdefault(f.owner, {})[f.id] = f.motor()
            for owner, motors in by_owner.items():
                await self.send(owner, {"type": "motors", "rt": rt_r, "rt_avg": round(self.rt_avg, 2),
                                        "can_spawn": block is None, "spawn_block": block,
                                        "flies": motors})
            for ws, watching in list(self.viewers.items()):
                f = self.flies.get(watching)
                if f is None or f.out is None:
                    continue
                await self.send(ws, {"type": "motor", "fly": f.id, "rt": rt_r,
                                     "t": round(f.brain.t * f.brain.dt, 1), **f.motor()})
                await self.send(ws, np.flatnonzero(f.delta).astype("<u4").tobytes())

            await asyncio.sleep(max(0.0, win - elapsed))


def explorer_data(brain, circ, mb, ws_port):
    """Static data for the explorer: positions, classes, types, circuits."""
    n = brain.neurons
    xyz = n[["x", "y", "z"]].to_numpy(np.float64)
    ok = np.isfinite(xyz).all(axis=1)
    center = np.nanmedian(xyz[ok], axis=0)
    xyz = np.where(ok[:, None], xyz - center, 0.0)
    xyz[:, 1] *= -1                                # in FlyWire y grows downwards
    pos = xyz.astype("<f4").tobytes()

    def codes(col, missing="?"):
        vals = n[col].fillna(missing).astype(str) if col in n else None
        if vals is None:
            return [missing], np.zeros(len(n), np.int64)
        cat = vals.astype("category")
        return list(cat.cat.categories), cat.cat.codes.to_numpy().astype(np.int64)

    classes, cls = codes("super_class")
    cls = np.where(ok, cls, 255).astype("u1")      # 255 = no position: not drawn
    types, typ = codes("cell_type")
    nts, nt = codes("top_nt")
    sides, side = codes("side")

    def grp(*arrays):
        return sorted(set(np.concatenate([np.asarray(a, np.int64) for a in arrays]).tolist()))

    groups = [
        {"name": "Eyes: LC4 and LPLC2", "role": "sense", "idx": grp(circ.eye["L"], circ.eye["R"])},
        {"name": "Antennae: fruit", "role": "sense",
         "idx": grp(circ.odor["fruity"]["L"], circ.odor["fruity"]["R"])},
        {"name": "Antennae: rotten", "role": "sense",
         "idx": grp(circ.odor["rotten"]["L"], circ.odor["rotten"]["R"])},
        {"name": "Legs: sugar", "role": "sense", "idx": grp(circ.sugar)},
        {"name": "Head: bristles and antennae", "role": "sense", "idx": grp(circ.touch["L"], circ.touch["R"])},
        {"name": "Antennae: player odors", "role": "sense", "idx": grp(circ.players.all_idx())},
        {"name": "Mushroom body: Kenyon cells", "role": "brain", "idx": grp(mb.kc)},
        {"name": "Dopamine: reward (PAM)", "role": "brain", "idx": grp(mb.reward_idx)},
        {"name": "Dopamine: punishment (PPL1)", "role": "brain", "idx": grp(mb.punish_idx)},
        {"name": "Mushroom body outputs (MBON)", "role": "brain", "idx": grp(mb.mbon)},
        {"name": "Fruit perception", "role": "brain", "idx": grp(circ.perceive_idx["fruity"])},
        {"name": "Rotten perception", "role": "brain", "idx": grp(circ.perceive_idx["rotten"])},
        {"name": "Giant Fiber", "role": "action", "idx": grp(circ.gf["L"], circ.gf["R"])},
        {"name": "Escape descending neurons", "role": "action",
         "idx": grp(circ.escape["L"], circ.escape["R"])},
        {"name": "MN9: proboscis", "role": "action", "idx": grp(circ.mn9)},
        {"name": "Grooming: descending neurons", "role": "action", "idx": grp(circ.groom)},
    ]
    groups = [g for g in groups if g["idx"]]
    meta = {"n": int(len(n)), "ws_port": ws_port, "classes": classes, "types": types,
            "nts": nts, "sides": sides, "groups": groups}
    return {
        "/meta.json": (json.dumps(meta).encode(), "application/json"),
        "/pos.bin": (pos, "application/octet-stream"),
        "/cls.bin": (cls.tobytes(), "application/octet-stream"),
        "/typ.bin": (typ.astype("<u2").tobytes(), "application/octet-stream"),
        "/nt.bin": (nt.astype("u1").tobytes(), "application/octet-stream"),
        "/side.bin": (side.astype("u1").tobytes(), "application/octet-stream"),
    }


def start_http(host, port, blobs):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                page = EXPLORER / "index.html"
                if not page.exists():
                    self.send_error(404, "explorer/index.html is missing from the project folder")
                    return
                body, ctype = page.read_bytes(), "text/html; charset=utf-8"
            elif path in blobs:
                body, ctype = blobs[path]
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):          # no log line for every request
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--http-port", type=int, default=8766, help="3D explorer port")
    ap.add_argument("--dt", type=float, default=0.2, help="simulation step in ms")
    ap.add_argument("--spawn-rt", type=float, default=1.5,
                    help="new flies only if the average real-time factor is at least this")
    ap.add_argument("--max-flies", type=int, default=32, help="absolute fly limit")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="spontaneous activity on the sensory neurons, in Hz (0 = off)")
    ap.add_argument("--shiu", action="store_true",
                    help="original Shiu model, without the connectome corrections")
    ap.add_argument("--adapt-b", type=float, default=None,
                    help="changes the adaptation (mV per spike) relative to the recommended model")
    ap.add_argument("--swap-sides", action="store_true",
                    help="swaps left/right if the fly reacts on the wrong side")
    args = ap.parse_args()

    t0 = time.perf_counter()
    params = {} if args.shiu else dict(RECOMMENDED)
    if args.adapt_b is not None:
        params["adapt_b"] = args.adapt_b
    template = FlyBrain(dt=args.dt, params=params)
    circ = FlyCircuits(template, args.swap_sides)
    circ.set_noise(args.noise)
    print(f"Connectome loaded in {time.perf_counter() - t0:.1f} s "
          f"(engine {template.engine}, dt = {args.dt} ms, noise = {args.noise} Hz, "
          f"model {'original Shiu' if args.shiu else 'recommended'})")
    print(f"   corrections: interneurons {template.ln_fixed}, KC->KC cut {template.kc_cut:,}, "
          f"slow neuromodulators {template.mod_silenced}, adaptation {template.p['adapt_b']} mV")
    lines, warnings = circ.report()
    for line in lines:
        print("  ", line)
    for w in warnings:
        print("   WARNING:", w)

    t0 = time.perf_counter()
    full = circ.calibrate_perception()
    print("   Odor perception scale: "
          + ", ".join(f"{k} = {v:.0f} Hz" for k, v in full.items())
          + f"  ({time.perf_counter() - t0:.1f} s)")
    print(f"   New flies as long as the average real-time factor stays above {args.spawn_rt}")

    t0 = time.perf_counter()
    mb = MushroomBody(template)
    print(f"   Memory (mushroom body): {len(mb.pos):,} learning KC->MBON synapses, "
          f"{(mb.comp == 1).sum()} reward and {(mb.comp == -1).sum()} punishment MBONs read "
          f"({time.perf_counter() - t0:.1f} s)")

    server = Server(template, circ, mb, args.spawn_rt, args.max_flies)
    start_http(args.host, args.http_port, explorer_data(template, circ, mb, args.port))
    async with serve(server.handler, args.host, args.port, max_size=2**22):
        print(f"Listening on ws://{args.host}:{args.port}  (Ctrl+C to stop)")
        print(f"3D explorer: http://{args.host}:{args.http_port}")
        await server.sim_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
