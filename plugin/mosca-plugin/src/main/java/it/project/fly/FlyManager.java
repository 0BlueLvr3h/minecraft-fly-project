package it.project.fly;

import com.google.gson.JsonObject;
import net.kyori.adventure.text.Component;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.entity.Mob;
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
 * All the flies in the game. Each fly has its own brain on the Python server; the senses
 * of all of them go out in a single message per tick and the commands come back the same way.
 */
public final class FlyManager {

    final JavaPlugin plugin;
    final BrainLink brain;
    final Settings cfg;
    final Lang lang;
    private final File logDir;

    private final Map<String, FlyController> flies = new LinkedHashMap<>();
    private final Map<UUID, String> byEntity = new HashMap<>();
    private int nextId = 1;

    private final Set<UUID> debugViewers = new HashSet<>();
    private DebugLog log;
    private File lastLogFile;

    public FlyManager(JavaPlugin plugin, BrainLink brain, Settings cfg, Lang lang, File logDir) {
        this.plugin = plugin;
        this.brain = brain;
        this.cfg = cfg;
        this.lang = lang;
        this.logDir = logDir;
        brain.onConnect(() -> new ArrayList<>(flies.keySet()));   // on connect: recreate the brains
    }

    // --------------------------------------------------------- births and deaths

    /** Why a new fly cannot be born. */
    public enum Refusal {
        OWN_LIMIT,   // the player already has their flies (spawn.max-per-player)
        WAIT,        // the server is still measuring the last one born: a few seconds
        PC_LIMIT,    // real-time factor below the threshold: the brain PC cannot handle more flies
        MAX          // the server's --max-flies has been reached
    }

    /** Can a fly be born for this player? null = yes, otherwise the reason. */
    public Refusal refusal(UUID owner) {
        if (owner != null && cfg.maxPerPlayer() > 0 && owned(owner) >= cfg.maxPerPlayer()) return Refusal.OWN_LIMIT;
        if (flies.isEmpty() || brain.canSpawn()) return null;
        String why = brain.spawnBlock();
        if ("wait".equals(why)) return Refusal.WAIT;
        if ("max".equals(why)) return Refusal.MAX;
        return Refusal.PC_LIMIT;
    }

    /** Explanation, in the configured language, of why a fly is not born (for chat and messages). */
    public String describe(Refusal r, UUID owner) {
        return switch (r) {
            case OWN_LIMIT -> {
                FlyController mine = owner != null ? ownedBy(owner) : null;
                yield lang.get("refusal.own_limit",
                        "which", mine != null ? lang.get("refusal.own_limit_which", "id", mine.id()) : "",
                        "max", cfg.maxPerPlayer());
            }
            case WAIT -> lang.get("refusal.wait");
            case MAX -> lang.get("refusal.max");
            case PC_LIMIT -> lang.get("refusal.pc_limit",
                    "rt", String.format(java.util.Locale.ROOT, "%.2f", brain.rtAvg()));
        };
    }

    /** How many living flies this player has spawned. */
    public int owned(UUID owner) {
        int n = 0;
        for (FlyController f : flies.values()) if (owner.equals(f.owner())) n++;
        return n;
    }

    public FlyController byId(String id) {
        return flies.get(id);
    }

    /** The fly bonded to this player (the one that likes them most), or null. */
    public FlyController bondedTo(UUID player) {
        for (FlyController f : flies.values()) if (player.equals(f.favorite())) return f;
        return null;
    }

    public FlyController ownedBy(UUID owner) {
        for (FlyController f : flies.values()) if (owner.equals(f.owner())) return f;
        return null;
    }

    /**
     * Spawns a fly owned by a player (owner may be null). If force is false it first checks
     * refusal(): per-player limit and PC headroom. Returns null if it cannot be born.
     */
    public FlyController spawn(Location loc, UUID owner, boolean force) {
        if (!force && refusal(owner) != null) return null;
        String id = String.valueOf(nextId++);
        FlyController fly = new FlyController(id, owner, this);
        fly.spawn(loc);
        flies.put(id, fly);
        byEntity.put(fly.entityId(), id);
        if (brain.isConnected()) brain.sendNow(message("spawn", id));
        else brain.start();                                   // the spawn is sent on connect
        return fly;
    }

    public void remove(FlyController fly) {
        fly.despawn();
        forget(fly, "remove");
    }

    public void removeAll() {
        for (FlyController fly : new ArrayList<>(flies.values())) remove(fly);
    }

    /** Called by the controller when the fly dies (Minecraft plays the animation). */
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
            for (Mob a : w.getEntitiesByClass(Mob.class)) {
                if (a.getPersistentDataContainer().has(MoscaPlugin.FLY_KEY, PersistentDataType.BYTE)) a.remove();
            }
        }
    }

    // --------------------------------------------------------- lookup

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

        // 1. the senses of all flies in a single message
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

        // 2. each one applies the commands of its own brain
        for (FlyController f : list) f.act();

        // 3. debug
        if (!debugViewers.isEmpty()) showDebug();
        if (log != null) for (FlyController f : list) f.writeLog(log);
    }

    // --------------------------------------------------------- debug

    /** Toggles debugging for a player (action bar + CSV log). */
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
                plugin.getLogger().warning("Cannot create the debug log: " + e.getMessage());
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
