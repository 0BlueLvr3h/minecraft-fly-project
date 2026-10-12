r"""
Server dei cervelli (WebSocket)
-------------------------------
Fa girare in tempo reale un cervello per ogni mosca in gioco. Tutti i cervelli
condividono lo stesso connettoma (in memoria una volta sola) e girano in parallelo
sui core del PC. La traduzione sensi/motori sta in circuits.py.

Serve anche l'esploratore 3D: http://127.0.0.1:8766 (pagina in explorer/index.html).

Nuove mosche: possono nascere finché il tempo reale medio resta sopra --spawn-rt
(di default 1.5). Se il PC non ce la fa, il server risponde "can_spawn": false.

Protocollo (JSON):

  plugin -> server
    {"type": "hello", "role": "body"}                 appena collegato
    {"type": "spawn", "fly": "3"}                     nasce la mosca 3: nuovo cervello
    {"type": "remove", "fly": "3"}  /  {"type": "death", "fly": "3"}
    {"type": "senses", "flies": {"3": {...sensi...}, "4": {...}}}   ogni tick
       sensi = {"threats": [{"angle", "dist", "speed"}], "odors": [{"odor", "angle", "dist"}],
                "taste": {"sweet": 0..1}, "touch": {"L": 0..1, "R": 0..1}}
       angle = gradi rispetto alla direzione della mosca, positivo = a sinistra

  server -> plugin, ogni finestra di simulazione (50 ms di cervello)
    {"type": "motors", "rt": 1.8, "can_spawn": true,
     "flies": {"3": {...comandi...}, ...}}
       comandi = {"looming", "touch", "smell", "gf", "takeoff", "escape", "perceive",
                  "mn9", "mn9_avg", "feeding", "groom", "grooming", "walk", "dn", "spikes"}

  Esploratore 3D (client "viewer"):
    -> {"type": "hello", "role": "viewer"}
    -> {"type": "watch", "fly": "3"}                  quale cervello guardare
    <- {"type": "flies", "list": ["3", "4"], "watching": "3"}
    <- {"type": "motor", ...comandi della mosca guardata..., "rt": ...}
    <- un messaggio BINARIO per finestra: indici (uint32) dei neuroni che hanno sparato
    -> {"type": "stim", "idx": [..] oppure "cell_type": "LC4", "hz": 150, "ms": 600}
    <- {"type": "stim_ok", "n": 104, "label": "LC4"}
    <- {"type": "alive"} / {"type": "death"}          la mosca guardata nasce / muore

Uso:
  python scripts\02_server.py                    (modello consigliato, vedi flysim.RECOMMENDED)
  python scripts\02_server.py --spawn-rt 1.2     (permette più mosche, con meno margine)
  python scripts\02_server.py --shiu             (modello di Shiu originale, senza correzioni)
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

WINDOW_MS = 50.0                  # = 1 tick di Minecraft
EXPLORER = Path(__file__).resolve().parent.parent / "explorer"
NO_SENSES = {"threats": [], "odors": [], "taste": {}, "touch": {}}


class Fly:
    """Un cervello con i suoi circuiti, i suoi sensi e i suoi stimoli dall'esploratore."""

    def __init__(self, fid, brain, circ, owner):
        self.id, self.brain, self.circ, self.owner = fid, brain, circ, owner
        self.senses = dict(NO_SENSES)
        self.extra = []                   # stimoli dall'esploratore: [idx, hz, fino_a_ms]
        self.out = None
        self.delta = None
        self.loom = {"L": 0.0, "R": 0.0}
        self.smell = {}

    def prepare(self):
        s = self.senses
        self.loom = self.circ.looming(s.get("threats", []))
        self.smell = self.circ.smell(s.get("odors", []))
        self.circ.apply(self.loom, self.smell, s.get("taste", {}), s.get("touch", {}))
        now = self.brain.t * self.brain.dt
        self.extra = [e for e in self.extra if e[2] > now]
        for idx, hz, _ in self.extra:
            self.brain.add_stimulus(idx, hz)

    def step(self):                       # gira in un thread: Numba lascia lavorare gli altri core
        before = self.brain.counts.copy()
        self.brain.run(WINDOW_MS)
        self.delta = self.brain.counts - before
        self.out = self.circ.decode(self.delta, WINDOW_MS)

    def motor(self):
        t = self.senses.get("touch", {})
        return {
            "looming": [round(self.loom["L"], 3), round(self.loom["R"], 3)],
            "touch": [round(float(t.get("L", 0)), 2), round(float(t.get("R", 0)), 2)],
            "smell": {k: [round(v["L"], 3), round(v["R"], 3)] for k, v in self.smell.items()},
            **self.out,
        }


class Server:
    def __init__(self, template, circ_template, spawn_rt, max_flies):
        self.template, self.circ_template = template, circ_template
        self.spawn_rt, self.max_flies = spawn_rt, max_flies
        self.flies = {}                   # id -> Fly
        self.clients = set()
        self.viewers = {}                 # ws -> id della mosca guardata (o None)
        self.rt_avg = 99.0
        self.seed = 1
        self.last_spawn = 0.0

    # ------------------------------------------------------------- capacità
    def spawn_block(self):
        """Perché non può nascere un'altra mosca: None (può), "wait", "max" o "rt"."""
        if not self.flies:
            return None
        if time.monotonic() - self.last_spawn <= 3.0:            # aspetta che l'ultima pesi sulla media
            return "wait"
        if len(self.flies) >= self.max_flies:
            return "max"
        if self.rt_avg < self.spawn_rt:
            return "rt"
        return None

    def can_spawn(self):
        return self.spawn_block() is None

    # ------------------------------------------------------------- mosche
    async def spawn(self, fid, owner):
        if fid in self.flies:
            return
        brain = self.template.sibling(seed=self.seed)
        self.seed += 1
        self.last_spawn = time.monotonic()
        self.flies[fid] = Fly(fid, brain, self.circ_template.sibling(brain), owner)
        print(f"  + mosca {fid} ({len(self.flies)} in tutto, tempo reale medio {self.rt_avg:.2f})")
        for ws, watching in list(self.viewers.items()):
            if watching is None or watching not in self.flies:
                self.viewers[ws] = fid
                await self.send(ws, {"type": "alive"})
        await self.tell_flies()

    async def drop(self, fid, died):
        fly = self.flies.pop(fid, None)
        if fly is None:
            return
        print(f"  - mosca {fid} {'morta' if died else 'rimossa'} ({len(self.flies)} in tutto)")
        for ws, watching in list(self.viewers.items()):
            if watching == fid:
                await self.send(ws, {"type": "death"})
        await self.tell_flies()

    # ------------------------------------------------------------- invii
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

    # ------------------------------------------------------------- connessioni
    async def handler(self, ws):
        self.clients.add(ws)
        print(f"+ client collegato ({len(self.clients)} totali)")
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
                    print("  (è un esploratore 3D)")
                    await self.tell_flies(only=ws)
                elif kind == "hello" and msg.get("role") == "body":
                    print("  (è il plugin di Minecraft)")
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
                await self.drop(fid, died=False)        # il plugin se n'è andato: niente mosche orfane
            print(f"- client scollegato ({len(self.clients)} totali)")

    async def stim(self, ws, msg):
        fly = self.flies.get(self.viewers.get(ws))
        if "cell_type" in msg:
            label = str(msg["cell_type"])
            idx = self.template.find(label)
        else:
            idx = np.asarray(msg.get("idx", []), np.int64)
            idx = idx[(idx >= 0) & (idx < self.template.N)]
            label = str(msg.get("label") or "neuroni scelti")
        if fly is None:
            await self.send(ws, {"type": "stim_ok", "n": 0, "label": "nessuna mosca viva"})
            return
        hz = float(min(max(msg.get("hz", 150.0), 0.0), 500.0))
        ms = float(min(max(msg.get("ms", 600.0), 50.0), 5000.0))
        if len(idx):
            fly.extra.append([idx, hz, fly.brain.t * fly.brain.dt + ms])
        await self.send(ws, {"type": "stim_ok", "n": int(len(idx)), "label": label})

    # ------------------------------------------------------------- ciclo
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
            self.rt_avg += (min(rt, 20.0) - self.rt_avg) * 0.05      # media di qualche secondo
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


def explorer_data(brain, circ, ws_port):
    """I dati statici per l'esploratore: posizioni, classi, tipi, circuiti."""
    n = brain.neurons
    xyz = n[["x", "y", "z"]].to_numpy(np.float64)
    ok = np.isfinite(xyz).all(axis=1)
    center = np.nanmedian(xyz[ok], axis=0)
    xyz = np.where(ok[:, None], xyz - center, 0.0)
    xyz[:, 1] *= -1                                # in FlyWire la y cresce verso il basso
    pos = xyz.astype("<f4").tobytes()

    def codes(col, missing="?"):
        vals = n[col].fillna(missing).astype(str) if col in n else None
        if vals is None:
            return [missing], np.zeros(len(n), np.int64)
        cat = vals.astype("category")
        return list(cat.cat.categories), cat.cat.codes.to_numpy().astype(np.int64)

    classes, cls = codes("super_class")
    cls = np.where(ok, cls, 255).astype("u1")      # 255 = senza posizione: non si disegna
    types, typ = codes("cell_type")
    nts, nt = codes("top_nt")
    sides, side = codes("side")

    def grp(*arrays):
        return sorted(set(np.concatenate([np.asarray(a, np.int64) for a in arrays]).tolist()))

    groups = [
        {"name": "Occhi: LC4 e LPLC2", "role": "senso", "idx": grp(circ.eye["L"], circ.eye["R"])},
        {"name": "Antenne: frutta", "role": "senso",
         "idx": grp(circ.odor["fruity"]["L"], circ.odor["fruity"]["R"])},
        {"name": "Antenne: marcio", "role": "senso",
         "idx": grp(circ.odor["rotten"]["L"], circ.odor["rotten"]["R"])},
        {"name": "Zampe: zucchero", "role": "senso", "idx": grp(circ.sugar)},
        {"name": "Testa: setole e antenne", "role": "senso", "idx": grp(circ.touch["L"], circ.touch["R"])},
        {"name": "Percezione della frutta", "role": "cervello", "idx": grp(circ.perceive_idx["fruity"])},
        {"name": "Percezione del marcio", "role": "cervello", "idx": grp(circ.perceive_idx["rotten"])},
        {"name": "Giant Fiber", "role": "azione", "idx": grp(circ.gf["L"], circ.gf["R"])},
        {"name": "Discendenti di fuga", "role": "azione",
         "idx": grp(circ.escape["L"], circ.escape["R"])},
        {"name": "MN9: proboscide", "role": "azione", "idx": grp(circ.mn9)},
        {"name": "Pulizia: discendenti", "role": "azione", "idx": grp(circ.groom)},
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
                    self.send_error(404, "Manca explorer/index.html nella cartella del progetto")
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

        def log_message(self, *args):          # niente log per ogni richiesta
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--http-port", type=int, default=8766, help="porta dell'esploratore 3D")
    ap.add_argument("--dt", type=float, default=0.2, help="passo di simulazione in ms")
    ap.add_argument("--spawn-rt", type=float, default=1.5,
                    help="nuove mosche solo se il tempo reale medio è almeno questo")
    ap.add_argument("--max-flies", type=int, default=32, help="limite assoluto di mosche")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="attività spontanea sui neuroni sensoriali, in Hz (0 = spenta)")
    ap.add_argument("--shiu", action="store_true",
                    help="modello di Shiu originale, senza le correzioni del connettoma")
    ap.add_argument("--adapt-b", type=float, default=None,
                    help="cambia l'adattamento (mV per spike) rispetto al modello consigliato")
    ap.add_argument("--swap-sides", action="store_true",
                    help="inverte sinistra/destra se la mosca reagisce dalla parte sbagliata")
    args = ap.parse_args()

    t0 = time.perf_counter()
    params = {} if args.shiu else dict(RECOMMENDED)
    if args.adapt_b is not None:
        params["adapt_b"] = args.adapt_b
    template = FlyBrain(dt=args.dt, params=params)
    circ = FlyCircuits(template, args.swap_sides)
    circ.set_noise(args.noise)
    print(f"Connettoma caricato in {time.perf_counter() - t0:.1f} s "
          f"(motore {template.engine}, dt = {args.dt} ms, rumore = {args.noise} Hz, "
          f"modello {'Shiu originale' if args.shiu else 'consigliato'})")
    print(f"   correzioni: interneuroni {template.ln_fixed}, KC->KC tagliate {template.kc_cut:,}, "
          f"neuromodulatori lenti {template.mod_silenced}, adattamento {template.p['adapt_b']} mV")
    lines, warnings = circ.report()
    for line in lines:
        print("  ", line)
    for w in warnings:
        print("   ATTENZIONE:", w)

    t0 = time.perf_counter()
    full = circ.calibrate_perception()
    print("   Scala percezione odori: "
          + ", ".join(f"{k} = {v:.0f} Hz" for k, v in full.items())
          + f"  ({time.perf_counter() - t0:.1f} s)")
    print(f"   Nuove mosche finché il tempo reale medio resta sopra {args.spawn_rt}")

    server = Server(template, circ, args.spawn_rt, args.max_flies)
    start_http(args.host, args.http_port, explorer_data(template, circ, args.port))
    async with serve(server.handler, args.host, args.port, max_size=2**22):
        print(f"In ascolto su ws://{args.host}:{args.port}  (Ctrl+C per fermare)")
        print(f"Esploratore 3D: http://{args.host}:{args.http_port}")
        await server.sim_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nFermato.")
