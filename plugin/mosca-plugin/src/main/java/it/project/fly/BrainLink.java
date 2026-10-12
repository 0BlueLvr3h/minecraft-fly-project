package it.project.fly;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.bukkit.Bukkit;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.scheduler.BukkitTask;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.WebSocket;
import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * Collegamento WebSocket con il server dei cervelli (02_server.py), uno per mosca.
 * Tutta la rete gira fuori dal main thread: il tick del server legge solo
 * l'ultimo comando ricevuto per ogni mosca e non aspetta mai il cervello.
 */
public final class BrainLink {

    /**
     * Un messaggio "motor" dal cervello.
     * fwd/left: direzione di fuga. walkFwd/walkTurn: camminata letta dai DN (solo debug).
     * perceive: quanto il cervello sente ogni odore (0..1).
     * memValence: cosa ricorda dell'odore di player che sente adesso (+ buono, - cattivo);
     * memFamiliar: quanto il corpo fungiforme sta sentendo quell'odore (0..1).
     */
    public record Motor(boolean takeoff, double fwd, double left,
                        double walkFwd, double walkTurn,
                        double loomL, double loomR, int gfL, int gfR, double rt,
                        Map<String, Double> perceive, int spikes,
                        double mn9, boolean feeding, double groom, boolean grooming,
                        double memValence, double memFamiliar, double playerSmell) {}

    private final JavaPlugin plugin;
    private final URI uri;
    private final HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(3)).build();

    private volatile WebSocket ws;
    private volatile boolean wanted;
    private volatile boolean connecting;
    private BukkitTask reconnectTask;
    private CompletableFuture<WebSocket> sending = CompletableFuture.completedFuture(null);

    private final Map<String, Motor> latest = new ConcurrentHashMap<>();
    private final Map<String, Motor> pendingTakeoff = new ConcurrentHashMap<>();
    private volatile boolean canSpawn = true;
    private volatile double rtAvg = 0;
    private volatile String spawnBlock = null;      // perché il server non accetta mosche: wait, rt, max
    private Supplier<List<String>> aliveFlies = List::of;

    public BrainLink(JavaPlugin plugin, String url) {
        this.plugin = plugin;
        this.uri = URI.create(url);
    }

    // ------------------------------------------------------------ connessione

    /** Inizia a collegarsi (e a ricollegarsi ogni 5 s se cade). */
    public void start() {
        wanted = true;
        if (ws == null && !connecting) connect();
        if (reconnectTask == null) {
            reconnectTask = Bukkit.getScheduler().runTaskTimerAsynchronously(plugin, () -> {
                if (wanted && ws == null && !connecting) connect();
            }, 100L, 100L);
        }
    }

    /** Al collegamento chiede la lista delle mosche vive, per ricreare i loro cervelli. */
    public void onConnect(Supplier<List<String>> aliveFlies) {
        this.aliveFlies = aliveFlies;
    }

    /** Si scollega: il server Python elimina i cervelli di questo plugin. */
    public void stop() {
        wanted = false;
        if (reconnectTask != null) {
            reconnectTask.cancel();
            reconnectTask = null;
        }
        WebSocket w = ws;
        ws = null;
        if (w != null) {
            try {
                w.sendClose(WebSocket.NORMAL_CLOSURE, "bye");
            } catch (Exception ignored) {
                w.abort();
            }
        }
        latest.clear();
        pendingTakeoff.clear();
        canSpawn = true;
        spawnBlock = null;
    }

    public boolean isConnected() {
        return ws != null;
    }

    private void connect() {
        connecting = true;
        http.newWebSocketBuilder()
                .buildAsync(uri, new Listener())
                .whenComplete((w, err) -> {
                    connecting = false;
                    if (err != null) {
                        plugin.getLogger().warning("Cervello non raggiungibile su " + uri
                                + " (riprovo tra 5 s)");
                    } else if (!wanted) {
                        w.abort();
                    } else {
                        ws = w;
                        plugin.getLogger().info("Collegato al cervello: " + uri);
                        // sul main thread: saluta e fa nascere i cervelli delle mosche già in gioco
                        Bukkit.getScheduler().runTask(plugin, () -> {
                            sendNow("{\"type\": \"hello\", \"role\": \"body\"}");
                            for (String id : aliveFlies.get()) {
                                sendNow("{\"type\": \"spawn\", \"fly\": \"" + id + "\"}");
                            }
                        });
                    }
                });
    }

    // ------------------------------------------------------------ invio

    /** Manda un messaggio importante: aspetta che parta quello precedente invece di saltarlo. */
    public synchronized void sendNow(String json) {
        WebSocket w = ws;
        if (w == null) return;
        try {
            sending.join();
            sending = w.sendText(json, true);
            sending.join();
        } catch (Exception ignored) {
        }
    }

    /** Manda un messaggio. Se il precedente non è ancora partito, salta questo. */
    public synchronized void send(String json) {
        WebSocket w = ws;
        if (w == null || !sending.isDone()) return;
        try {
            sending = w.sendText(json, true);
        } catch (Exception e) {
            ws = null;
        }
    }

    // ------------------------------------------------------------ ricezione

    /** Ultimo comando motorio ricevuto per una mosca (può essere null). */
    public Motor latest(String fly) {
        return latest.get(fly);
    }

    /** Restituisce un decollo in attesa per una mosca, se c'è, e lo consuma. */
    public Motor pollTakeoff(String fly) {
        return pendingTakeoff.remove(fly);
    }

    /** Il server ha margine per un altro cervello? (tempo reale medio sopra la soglia) */
    public boolean canSpawn() {
        return canSpawn;
    }

    /** Perché il server non accetta nuove mosche: "wait", "rt", "max", oppure null. */
    public String spawnBlock() {
        return canSpawn ? null : spawnBlock;
    }

    public double rtAvg() {
        return rtAvg;
    }

    /** Dimentica i comandi di una mosca che non c'è più. */
    public void forget(String fly) {
        latest.remove(fly);
        pendingTakeoff.remove(fly);
    }

    private void handle(String raw) {
        try {
            JsonObject batch = JsonParser.parseString(raw).getAsJsonObject();
            if (!batch.has("type") || !"motors".equals(batch.get("type").getAsString())) return;
            canSpawn = batch.get("can_spawn").getAsBoolean();
            rtAvg = batch.get("rt_avg").getAsDouble();
            spawnBlock = batch.has("spawn_block") && !batch.get("spawn_block").isJsonNull()
                    ? batch.get("spawn_block").getAsString() : null;
            double rt = batch.get("rt").getAsDouble();
            JsonObject flies = batch.getAsJsonObject("flies");
            for (String id : flies.keySet()) {
                Motor m = parse(flies.getAsJsonObject(id), rt);
                latest.put(id, m);
                if (m.takeoff()) pendingTakeoff.put(id, m);
            }
        } catch (Exception e) {
            plugin.getLogger().fine("Messaggio dal cervello non valido: " + e.getMessage());
        }
    }

    private static Motor parse(JsonObject o, double rt) {
        JsonArray loom = o.getAsJsonArray("looming");
        JsonArray gf = o.getAsJsonArray("gf");
        JsonObject esc = o.getAsJsonObject("escape");
        JsonObject walk = o.has("walk") ? o.getAsJsonObject("walk") : null;
        JsonObject mem = o.has("memory") ? o.getAsJsonObject("memory") : null;
        Map<String, Double> perceive = new HashMap<>();
        if (o.has("perceive")) {
            JsonObject p = o.getAsJsonObject("perceive");
            for (String k : p.keySet()) perceive.put(k, p.get(k).getAsDouble());
        }
        Motor m = new Motor(
                o.get("takeoff").getAsBoolean(),
                esc.get("fwd").getAsDouble(), esc.get("left").getAsDouble(),
                walk != null ? walk.get("fwd").getAsDouble() : 0.0,
                walk != null ? walk.get("turn").getAsDouble() : 0.0,
                loom.get(0).getAsDouble(), loom.get(1).getAsDouble(),
                gf.get(0).getAsInt(), gf.get(1).getAsInt(),
                rt,
                perceive,
                o.has("spikes") ? o.get("spikes").getAsInt() : -1,
                o.has("mn9") ? o.get("mn9").getAsDouble() : 0.0,
                o.has("feeding") && o.get("feeding").getAsBoolean(),
                o.has("groom") ? o.get("groom").getAsDouble() : 0.0,
                o.has("grooming") && o.get("grooming").getAsBoolean(),
                mem != null ? mem.get("valence").getAsDouble() : 0.0,
                mem != null ? mem.get("familiar").getAsDouble() : 0.0,
                o.has("player_smell") ? o.get("player_smell").getAsDouble() : 0.0);
        return m;
    }

    private final class Listener implements WebSocket.Listener {
        private final StringBuilder buf = new StringBuilder();

        @Override
        public void onOpen(WebSocket w) {
            w.request(1);
        }

        @Override
        public CompletionStage<?> onText(WebSocket w, CharSequence data, boolean last) {
            buf.append(data);
            if (last) {
                handle(buf.toString());
                buf.setLength(0);
            }
            w.request(1);
            return null;
        }

        @Override
        public CompletionStage<?> onClose(WebSocket w, int code, String reason) {
            if (ws == w) {
                ws = null;
                plugin.getLogger().info("Cervello scollegato");
            }
            return null;
        }

        @Override
        public void onError(WebSocket w, Throwable error) {
            if (ws == w) {
                ws = null;
                plugin.getLogger().warning("Errore di collegamento col cervello: " + error.getMessage());
            }
        }
    }
}
