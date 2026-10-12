package it.project.fly;

import com.google.gson.JsonObject;
import net.kyori.adventure.text.Component;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.entity.Allay;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.persistence.PersistentDataType;
import org.bukkit.plugin.java.JavaPlugin;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

/**
 * Tutte le mosche in gioco. Ogni mosca ha il suo cervello sul server Python; i sensi
 * di tutte partono in un solo messaggio per tick e i comandi tornano allo stesso modo.
 */
public final class FlyManager {

    final JavaPlugin plugin;
    final BrainLink brain;
    final Settings cfg;
    private final File logDir;

    private final Map<String, FlyController> flies = new LinkedHashMap<>();
    private final Map<UUID, String> byEntity = new HashMap<>();
    private int nextId = 1;

    private final Set<UUID> debugViewers = new HashSet<>();
    private DebugLog log;
    private File lastLogFile;

    public FlyManager(JavaPlugin plugin, BrainLink brain, Settings cfg, File logDir) {
        this.plugin = plugin;
        this.brain = brain;
        this.cfg = cfg;
        this.logDir = logDir;
        brain.onConnect(() -> new ArrayList<>(flies.keySet()));   // al collegamento: ricrea i cervelli
    }

    // --------------------------------------------------------- nascite e morti

    /** Perché una nuova mosca non può nascere. */
    public enum Refusal {
        OWN_LIMIT,   // il player ha già le sue mosche (spawn.max-per-player)
        WAIT,        // il server sta ancora misurando l'ultima nata: qualche secondo
        PC_LIMIT,    // tempo reale sotto la soglia: il PC dei cervelli non regge altre mosche
        MAX          // raggiunto --max-flies del server
    }

    /** Può nascere una mosca per questo player? null = sì, altrimenti il motivo. */
    public Refusal refusal(UUID owner) {
        if (owner != null && cfg.maxPerPlayer() > 0 && owned(owner) >= cfg.maxPerPlayer()) return Refusal.OWN_LIMIT;
        if (flies.isEmpty() || brain.canSpawn()) return null;
        String why = brain.spawnBlock();
        if ("wait".equals(why)) return Refusal.WAIT;
        if ("max".equals(why)) return Refusal.MAX;
        return Refusal.PC_LIMIT;
    }

    /** Spiegazione in italiano del perché una mosca non nasce (per chat e messaggi). */
    public String describe(Refusal r, UUID owner) {
        return switch (r) {
            case OWN_LIMIT -> {
                FlyController mine = owner != null ? ownedBy(owner) : null;
                yield "hai già la tua mosca" + (mine != null ? " (mosca " + mine.id() + ")" : "")
                        + ", massimo " + cfg.maxPerPlayer() + " per player.";
            }
            case WAIT -> "il PC sta ancora misurando l'ultima mosca nata, riprova tra qualche secondo.";
            case MAX -> "raggiunto il numero massimo di mosche del server dei cervelli.";
            case PC_LIMIT -> String.format(java.util.Locale.ROOT,
                    "il PC dei cervelli è al limite (tempo reale %.2f).", brain.rtAvg());
        };
    }

    /** Quante mosche vive ha fatto nascere questo player. */
    public int owned(UUID owner) {
        int n = 0;
        for (FlyController f : flies.values()) if (owner.equals(f.owner())) n++;
        return n;
    }

    public FlyController ownedBy(UUID owner) {
        for (FlyController f : flies.values()) if (owner.equals(f.owner())) return f;
        return null;
    }

    /**
     * Fa nascere una mosca di un player (owner può essere null). Se force è falso controlla
     * prima refusal(): limite per player e margine del PC. Restituisce null se non può nascere.
     */
    public FlyController spawn(Location loc, UUID owner, boolean force) {
        if (!force && refusal(owner) != null) return null;
        String id = String.valueOf(nextId++);
        FlyController fly = new FlyController(id, owner, this);
        fly.spawn(loc);
        flies.put(id, fly);
        byEntity.put(fly.entityId(), id);
        if (brain.isConnected()) brain.sendNow(message("spawn", id));
        else brain.start();                                   // al collegamento manda lo spawn
        return fly;
    }

    public void remove(FlyController fly) {
        fly.despawn();
        forget(fly, "remove");
    }

    public void removeAll() {
        for (FlyController fly : new ArrayList<>(flies.values())) remove(fly);
    }

    /** Chiamato dal controller quando la mosca muore (l'animazione la fa Minecraft). */
    void died(FlyController fly) {
        forget(fly, "death");
    }

    private void forget(FlyController fly, String why) {
        flies.remove(fly.id());
        byEntity.values().remove(fly.id());
        brain.forget(fly.id());
        if (brain.isConnected()) brain.sendNow(message(why, fly.id()));
        if (flies.isEmpty()) {
            brain.stop();
            closeLog();
        }
    }

    private static String message(String type, String id) {
        JsonObject m = new JsonObject();
        m.addProperty("type", type);
        m.addProperty("fly", id);
        return m.toString();
    }

    public void removeLeftovers() {
        for (World w : Bukkit.getWorlds()) {
            for (Allay a : w.getEntitiesByClass(Allay.class)) {
                if (a.getPersistentDataContainer().has(MoscaPlugin.FLY_KEY, PersistentDataType.BYTE)) a.remove();
            }
        }
    }

    // --------------------------------------------------------- ricerca

    public boolean isFly(Entity e) {
        return e.getPersistentDataContainer().has(MoscaPlugin.FLY_KEY, PersistentDataType.BYTE);
    }

    public FlyController get(Entity e) {
        String id = byEntity.get(e.getUniqueId());
        return id == null ? null : flies.get(id);
    }

    public FlyController nearest(Location loc) {
        FlyController best = null;
        double bestD = Double.MAX_VALUE;
        for (FlyController f : flies.values()) {
            Location at = f.location();
            if (at == null || at.getWorld() != loc.getWorld()) continue;
            double d = at.distanceSquared(loc);
            if (d < bestD) {
                bestD = d;
                best = f;
            }
        }
        return best;
    }

    public Collection<FlyController> all() {
        return flies.values();
    }

    public int count() {
        return flies.size();
    }

    // --------------------------------------------------------- tick

    public void tick() {
        if (flies.isEmpty()) return;
        List<FlyController> list = new ArrayList<>(flies.values());

        // 1. i sensi di tutte le mosche in un solo messaggio
        JsonObject all = new JsonObject();
        for (FlyController f : list) {
            JsonObject s = f.senses();
            if (s != null) all.add(f.id(), s);
        }
        if (brain.isConnected()) {
            JsonObject msg = new JsonObject();
            msg.addProperty("type", "senses");
            msg.add("flies", all);
            brain.send(msg.toString());
        }

        // 2. ognuna applica i comandi del suo cervello
        for (FlyController f : list) f.act();

        // 3. debug
        if (!debugViewers.isEmpty()) showDebug();
        if (log != null) for (FlyController f : list) f.writeLog(log);
    }

    // --------------------------------------------------------- debug

    /** Attiva/disattiva il debug per un player (action bar + registro CSV). */
    public boolean toggleDebug(Player p) {
        if (debugViewers.remove(p.getUniqueId())) {
            if (debugViewers.isEmpty()) closeLog();
            return false;
        }
        debugViewers.add(p.getUniqueId());
        if (log == null) {
            try {
                log = new DebugLog(logDir);
                lastLogFile = log.file();
            } catch (IOException e) {
                plugin.getLogger().warning("Impossibile creare il registro di debug: " + e.getMessage());
            }
        }
        return true;
    }

    public File logFile() {
        return lastLogFile;
    }

    public void closeLog() {
        if (log != null) {
            log.close();
            log = null;
        }
    }

    private void showDebug() {
        for (UUID id : debugViewers) {
            Player p = Bukkit.getPlayer(id);
            if (p == null) continue;
            FlyController f = nearest(p.getLocation());
            if (f != null) p.sendActionBar(Component.text(f.debugLine(p)));
        }
    }
}
