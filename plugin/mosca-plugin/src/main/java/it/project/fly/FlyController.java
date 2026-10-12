package it.project.fly;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.format.NamedTextColor;
import org.bukkit.attribute.Attribute;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.Bukkit;
import org.bukkit.GameMode;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.Particle;
import org.bukkit.block.Block;
import org.bukkit.entity.Allay;
import org.bukkit.entity.Item;
import org.bukkit.entity.Player;
import org.bukkit.inventory.ItemStack;
import org.bukkit.persistence.PersistentDataType;
import org.bukkit.util.Vector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

/**
 * Una mosca (un Allay) con il suo cervello. A ogni tick il FlyManager:
 *   1. le chiede i sensi (player che si avvicinano, odori, gusto, tatto): senses()
 *   2. dopo averli mandati al server, le fa applicare i comandi del suo cervello: act()
 * Con l'AI spenta l'Allay ignora la velocità, quindi lo spostiamo noi con un teleport.
 */
public final class FlyController {

    public enum State { GROUND, FLYING }

    // Fisica del volo di fuga (blocchi per tick)
    private static final double TAKEOFF_SPEED = 0.6;
    private static final double TAKEOFF_LIFT = 0.35;
    private static final double DRAG = 0.9;
    private static final double SINK = 0.025;
    private static final int MAX_FLIGHT_TICKS = 40;
    private static final int GROUND_SCAN = 4;        // quanti blocchi sotto cerca il terreno
    private static final double TASTE_DIST = 1.0;    // blocchi: "ci sono sopra", può assaggiare
    private static final int BITE_TICKS = 60;        // ogni 3 s di pasto consuma un item

    private final String id;
    private final UUID owner;                        // player che l'ha fatta nascere (null = nessuno)
    private final FlyManager mgr;
    private final JavaPlugin plugin;
    private final BrainLink brain;
    private final Settings cfg;
    private final Spinal spinal;
    private final List<Spinal.Source> sources = new ArrayList<>();
    private boolean blocked;
    private String mode = "ferma";

    private Allay fly;
    private State state = State.GROUND;
    private Vector vel = new Vector();
    private int flightTicks;
    private double baseY;
    private double walkFwd, walkTurn;                // comandi di camminata smussati
    private long tick;
    private int takeoffs;
    private final Map<UUID, Vector> lastPos = new HashMap<>();     // ultima posizione di ogni player
    private final Map<UUID, double[]> lastSeen = new HashMap<>();   // debug: dist, velocità, angolo
    private final Map<UUID, String> heldOdor = new HashMap<>();     // debug: odore in mano
    private boolean tookOff;
    private boolean reflexTakeoff;                   // decollo dal riflesso del corpo (non dal cervello)
    private double lastHitAngle;
    private double taste;                            // zucchero sotto le zampe (0..1)
    private Item tasteItem;                          // item a terra che sta assaggiando
    private int feedTicks;
    private double touchL, touchR;                   // colpo sulla testa, per lato (0..1), si attenua
    private int groomTicks;
    private float groomYaw;

    FlyController(String id, UUID owner, FlyManager mgr) {
        this.id = id;
        this.owner = owner;
        this.mgr = mgr;
        this.plugin = mgr.plugin;
        this.brain = mgr.brain;
        this.cfg = mgr.cfg;
        this.spinal = new Spinal(cfg);
    }

    public String id() { return id; }
    public UUID owner() { return owner; }
    public UUID entityId() { return fly != null ? fly.getUniqueId() : null; }
    public Location location() { return fly != null && !fly.isDead() ? fly.getLocation() : null; }
    public double health() { return fly != null ? fly.getHealth() : 0; }

    // --------------------------------------------------------- ciclo di vita

    void spawn(Location loc) {
        fly = loc.getWorld().spawn(loc, Allay.class, a -> {
            a.setAI(false);
            a.setGravity(false);
            a.setCanPickupItems(false);
            a.setPersistent(false);          // non viene salvata: al riavvio sparisce
            a.setRemoveWhenFarAway(false);
            a.setSilent(true);
            a.customName(Component.text("Mosca " + id));      // i cuori arrivano con refreshName()
            a.setCustomNameVisible(true);
            a.getPersistentDataContainer().set(MoscaPlugin.FLY_KEY, PersistentDataType.BYTE, (byte) 1);
        });
        baseY = loc.getY();
        state = State.GROUND;
        refreshName();
        walkFwd = walkTurn = 0;
        takeoffs = 0;
        lastPos.clear();
    }

    /** Toglie l'entità dal mondo (il manager avvisa il server). */
    void despawn() {
        if (fly != null) {
            fly.remove();
            fly = null;
        }
        lastPos.clear();
    }

    /** Morta: avvisa il cervello (che si spegne) e lascia a Minecraft l'animazione. */
    public void onDeath() {
        if (fly == null) return;
        Location loc = fly.getLocation();
        for (Player p : loc.getWorld().getPlayers()) {
            if (p.getLocation().distance(loc) < 48) {
                p.sendMessage(Component.text("La mosca " + id + " è morta.", NamedTextColor.GRAY));
            }
        }
        fly = null;                                  // niente remove(): l'animazione di morte resta
        lastPos.clear();
        mgr.died(this);                              // il manager avvisa il server: il cervello si ferma
    }

    /** Nome con i cuori: rossi quelli pieni, grigi quelli persi. */
    public void refreshName() {
        if (fly == null) return;
        Bukkit.getScheduler().runTask(plugin, () -> {   // il danno viene applicato dopo l'evento
            if (fly == null || fly.isDead()) return;
            double max = fly.getAttribute(Attribute.MAX_HEALTH) != null
                    ? fly.getAttribute(Attribute.MAX_HEALTH).getValue() : 20.0;
            int full = (int) Math.ceil(fly.getHealth() / 2.0);
            int total = (int) Math.ceil(max / 2.0);
            fly.customName(Component.text("Mosca " + id + " ")
                    .append(Component.text("❤".repeat(Math.max(0, full)), NamedTextColor.RED))
                    .append(Component.text("❤".repeat(Math.max(0, total - full)), NamedTextColor.DARK_GRAY)));
        });
    }

    public boolean exists() {
        return fly != null && !fly.isDead();
    }

    public State state() { return state; }
    public int takeoffs() { return takeoffs; }
    public double walkFwd() { return walkFwd; }
    public double walkTurn() { return walkTurn; }
    public String mode() { return state == State.FLYING ? "in fuga" : mode; }

    // --------------------------------------------------------------- tick

    /** Primo mezzo tick: cosa sente la mosca (null se non è nel mondo in questo momento). */
    JsonObject senses() {
        if (fly == null || fly.isDead() || !fly.isValid()) return null;   // chunk non caricato: aspettiamo
        return collectSenses(fly.getLocation());
    }

    /** Secondo mezzo tick: applica i comandi del cervello. */
    void act() {
        if (fly == null || fly.isDead() || !fly.isValid()) return;
        tick++;
        Location loc = fly.getLocation();

        BrainLink.Motor takeoff = brain.pollTakeoff(id);   // consumato anche in volo
        tookOff = reflexTakeoff;
        reflexTakeoff = false;
        if (state == State.GROUND) {
            if (takeoff != null) {
                startFlight(loc, takeoff);
                tookOff = true;
            } else {
                walk(loc);
            }
        } else {
            flightStep(loc);
        }
    }

    void writeLog(DebugLog log) {
        if (fly == null || fly.isDead()) return;
        BrainLink.Motor m = brain.latest(id);
        Location at = fly.getLocation();
        String base = String.format(Locale.ROOT, "%s,%d,%d,%s,%.2f,%.2f,%.2f,%.1f",
                id, log.elapsed(), tick, mode().replace(',', ' '), at.getX(), at.getY(), at.getZ(), at.getYaw());
        String tastePart = String.format(Locale.ROOT, ",%.2f,%.1f,%d,%.2f,%.2f,%.1f,%d,%.1f",
                taste, m == null ? 0.0 : m.mn9(), m != null && m.feeding() ? 1 : 0,
                touchL, touchR, m == null ? 0.0 : m.groom(), m != null && m.grooming() ? 1 : 0,
                fly.getHealth());
        String brainPart = m == null
                ? "0,,,,,,,,,,,,,"
                : String.format(Locale.ROOT, "1,%.2f,%d,%.3f,%.3f,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f",
                        m.rt(), m.spikes(), m.loomL(), m.loomR(), m.gfL(), m.gfR(), tookOff ? 1 : 0,
                        m.fwd(), m.left(), walkFwd, walkTurn,
                        m.perceive().getOrDefault("fruity", 0.0), m.perceive().getOrDefault("rotten", 0.0));
        if (lastSeen.isEmpty()) {
            log.row(base + "," + brainPart + ",,,,," + tastePart);
            return;
        }
        for (Map.Entry<UUID, double[]> e : lastSeen.entrySet()) {
            Player p = Bukkit.getPlayer(e.getKey());
            String name = p != null ? p.getName() : e.getKey().toString();
            double[] s = e.getValue();
            log.row(base + "," + brainPart + String.format(Locale.ROOT, ",%s,%.2f,%.2f,%.1f,%s",
                    name, s[0], s[1], s[2], heldOdor.getOrDefault(e.getKey(), "")) + tastePart);
        }
    }

    /** Riga dell'action bar di debug per un player (si riferisce a questa mosca). */
    String debugLine(Player p) {
        BrainLink.Motor m = brain.latest(id);
        double[] seen = lastSeen.get(p.getUniqueId());
        String you = seen == null ? "fuori portata"
                : String.format(Locale.ROOT, "%.1f blocchi, %+.1f m/s", seen[0], seen[1]);
        return m == null
                ? "Mosca " + id + ": cervello non collegato | " + you + " | " + mode()
                : String.format(Locale.ROOT,
                        "Mosca %s | tu: %s | vita %.0f | looming %.2f/%.2f | GF %d/%d | MN9 %.0f | %s | spike %d | rt %.2f",
                        id, you, health(), m.loomL(), m.loomR(), m.gfL(), m.gfR(), m.mn9(), mode(),
                        m.spikes(), m.rt());
    }

    // ---------------------------------------------------------------- sensi

    private JsonObject collectSenses(Location loc) {
        sources.clear();
        taste = 0.0;
        tasteItem = null;
        Vector f = forward(loc.getYaw());
        Vector l = left(f);
        Vector here = loc.toVector();
        double range = cfg.range();

        JsonArray threats = new JsonArray();
        JsonArray odors = new JsonArray();
        Set<UUID> seen = new HashSet<>();

        for (Player p : loc.getWorld().getPlayers()) {
            if (p.getGameMode() == GameMode.SPECTATOR) continue;
            Vector d = p.getLocation().add(0, 0.9, 0).toVector().subtract(here);
            double dist = d.length();
            if (dist > range) continue;
            double angle = angleOf(d, f, l);

            // occhi: quanto in fretta il PLAYER si muove verso la mosca.
            // Contiamo solo il movimento del player, non quello della mosca: quando è lei
            // ad andare verso di te (per la mela) non deve spaventarsi da sola.
            seen.add(p.getUniqueId());
            Vector pos = p.getLocation().toVector();
            Vector prev = lastPos.put(p.getUniqueId(), pos);
            double speed = 0.0;
            if (prev != null && dist > 1e-6) {
                Vector vel = pos.clone().subtract(prev).multiply(20.0);    // blocchi al secondo
                Vector towardFly = d.clone().multiply(-1.0 / dist);        // dal player alla mosca
                speed = clamp(vel.dot(towardFly), -20, 20);
            }
            lastSeen.put(p.getUniqueId(), new double[]{dist, speed, angle});
            JsonObject t = new JsonObject();
            t.addProperty("angle", round(angle));
            t.addProperty("dist", round(dist));
            t.addProperty("speed", round(speed));
            threats.add(t);

            // antenne: l'odore dell'item che ha in mano
            String held = addOdor(odors, p.getInventory().getItemInMainHand(), angle, dist);
            String off = addOdor(odors, p.getInventory().getItemInOffHand(), angle, dist);
            if (dist <= TASTE_DIST + 0.9 && ("fruity".equals(held) || "fruity".equals(off))) {
                taste = 1.0;                         // è sulla mano con la frutta
            }
            heldOdor.put(p.getUniqueId(), held != null ? held : (off != null ? off : ""));
        }
        lastPos.keySet().retainAll(seen);
        lastSeen.keySet().retainAll(seen);
        heldOdor.keySet().retainAll(seen);

        // antenne: item lasciati a terra
        for (Item item : loc.getNearbyEntitiesByType(Item.class, range)) {
            Vector d = item.getLocation().toVector().subtract(here);
            String odor = addOdor(odors, item.getItemStack(), angleOf(d, f, l), d.length());
            if ("fruity".equals(odor) && d.length() <= TASTE_DIST + cfg.hoverHeight()) {
                taste = 1.0;                         // ci è sopra: le zampe toccano lo zucchero
                tasteItem = item;
            }
        }

        JsonObject msg = new JsonObject();
        msg.add("threats", threats);
        msg.add("odors", odors);
        JsonObject tasteObj = new JsonObject();
        tasteObj.addProperty("sweet", taste);
        msg.add("taste", tasteObj);
        JsonObject touchObj = new JsonObject();
        touchObj.addProperty("L", round(touchL));
        touchObj.addProperty("R", round(touchR));
        msg.add("touch", touchObj);
        touchL *= 0.8;                               // il tocco svanisce in circa 0,3 s
        touchR *= 0.8;
        if (touchL < 0.02) touchL = 0;
        if (touchR < 0.02) touchR = 0;
        return msg;
    }

    /** Aggiunge l'odore dell'item, se ne ha uno; restituisce il nome dell'odore o null. */
    private String addOdor(JsonArray odors, ItemStack stack, double angle, double dist) {
        if (stack == null || stack.getType() == Material.AIR) return null;
        String odor = cfg.odors().get(stack.getType());
        if (odor == null) return null;
        JsonObject o = new JsonObject();
        o.addProperty("odor", odor);
        o.addProperty("angle", round(angle));
        o.addProperty("dist", round(dist));
        odors.add(o);
        sources.add(new Spinal.Source(odor, angle, dist));    // per il midollo
        return odor;
    }

    // --------------------------------------------------------------- moto

    /** A terra (sospesa poco sopra): il midollo genera la camminata, pesata da ciò che il cervello percepisce. */
    private void walk(Location loc) {
        BrainLink.Motor m = brain.latest(id);
        if (m != null && m.grooming()) {
            groom(loc);
            return;
        }
        groomTicks = 0;
        if (m != null && m.feeding() && taste > 0) {
            eat(loc);
            return;
        }
        if (taste <= 0) feedTicks = 0;               // il pasto ricomincia solo se si allontana
        Spinal.Drive d = spinal.step(m != null ? m.perceive() : null, sources, blocked);
        mode = d.mode();
        double a = cfg.smoothing();
        walkFwd += (d.fwd() - walkFwd) * a;
        walkTurn += (d.turn() - walkTurn) * a;

        // sterzata positiva = sinistra = yaw che diminuisce
        float yaw = loc.getYaw() - (float) (walkTurn * cfg.maxTurn() / 20.0);
        Vector step = forward(yaw).multiply(walkFwd * cfg.maxSpeed() / 20.0);

        Location next = loc.clone().add(step);
        blocked = false;
        if (!free(next)) {
            Location up = next.clone().add(0, 1, 0);       // gradino: prova a salire
            if (free(up) && free(loc.clone().add(0, 1, 0))) {
                next = up;
                baseY += 1;
            } else {
                next = loc.clone();                         // muro: solo rotazione
                blocked = step.lengthSquared() > 1e-6;
            }
        }

        baseY += (groundTarget(next) - baseY) * 0.2;
        next.setY(baseY + 0.05 * Math.sin(tick * 0.15));
        next.setYaw(yaw);
        next.setPitch(0);
        fly.teleport(next);
    }

    /** Applica il tatto sul lato da cui arriva lo stimolo. Restituisce la direzione dello stimolo. */
    private Vector feel(Location from, double strength) {
        Location loc = fly.getLocation();
        Vector f = forward(loc.getYaw());
        Vector l = left(f);
        Vector d = from.toVector().subtract(loc.toVector());
        double angle = angleOf(d, f, l);
        if (angle > -20) touchL = Math.max(touchL, strength);
        if (angle < 20) touchR = Math.max(touchR, strength);
        lastHitAngle = angle;
        return d;
    }

    /**
     * Colpita (pugno, spada, freccia...). Il colpo diventa tatto sulle setole della testa dal
     * lato colpito, più forte quanto più è forte il colpo; se supera la soglia scatta anche il
     * riflesso del corpo e salta via. Il danno lo applica Minecraft (lampo rosso compreso).
     */
    public void hit(Location from, double damage) {
        if (fly == null || !fly.isValid()) return;
        Location loc = fly.getLocation();
        double strength = clamp(damage / 6.0, 0.3, 1.0);   // pugno ~0,3, spada ~1
        Vector d = feel(from, strength);
        Vector f = forward(loc.getYaw());

        if (cfg.hitReflex() > 0 && strength >= cfg.hitReflex() && state == State.GROUND) {
            // riflesso: via dal lato colpito
            Vector away = d.clone().setY(0).multiply(-1);
            if (away.lengthSquared() < 1e-6) away = f.clone().multiply(-1);
            away.normalize();
            vel = away.multiply(TAKEOFF_SPEED).setY(TAKEOFF_LIFT);
            state = State.FLYING;
            flightTicks = 0;
            takeoffs++;
            reflexTakeoff = true;
            loc.getWorld().spawnParticle(Particle.CLOUD, loc, 6, 0.15, 0.1, 0.15, 0.02);
        }
    }

    /** Discendenti delle setole accesi: la mosca si ferma e si strofina. */
    private void groom(Location loc) {
        mode = "si pulisce";
        walkFwd = 0;
        walkTurn = 0;
        if (groomTicks++ == 0) groomYaw = loc.getYaw();
        if (groomTicks % 4 == 1) fly.swingMainHand();             // le "zampe" che strofinano
        if (groomTicks % 8 == 1) {
            loc.getWorld().spawnParticle(Particle.WHITE_ASH, loc.clone().add(0, 0.5, 0), 4, 0.15, 0.1, 0.15, 0);
        }
        Location next = loc.clone();
        baseY += (groundTarget(next) - baseY) * 0.2;
        next.setY(baseY + 0.02 * Math.sin(groomTicks * 2.1));
        // piccoli scatti di testa: le zampe anteriori che strofinano il punto toccato
        next.setYaw(groomYaw + (float) (Math.sin(groomTicks * 1.7) * 5.0));
        fly.teleport(next);
    }

    /** MN9 acceso: proboscide fuori, la mosca resta ferma e mangia. */
    private void eat(Location loc) {
        mode = "mangia";
        walkFwd = 0;
        walkTurn = 0;
        feedTicks++;
        Location next = loc.clone();
        baseY += (groundTarget(next) - baseY) * 0.2;
        next.setY(baseY + 0.02 * Math.sin(tick * 0.5));         // piccoli movimenti mentre mangia
        fly.teleport(next);
        if (feedTicks % 10 == 0) {
            loc.getWorld().spawnParticle(Particle.HEART, loc.clone().add(0, 0.6, 0), 1, 0.1, 0.1, 0.1, 0);
        }
        if (feedTicks % BITE_TICKS == 0 && cfg.healPerBite() > 0) {   // mangiare ridà vita
            double max = fly.getAttribute(Attribute.MAX_HEALTH) != null
                    ? fly.getAttribute(Attribute.MAX_HEALTH).getValue() : 20.0;
            if (fly.getHealth() < max) {
                fly.setHealth(Math.min(max, fly.getHealth() + cfg.healPerBite()));
                refreshName();
            }
        }
        if (tasteItem != null && feedTicks % BITE_TICKS == 0) {   // un morso alla volta
            ItemStack stack = tasteItem.getItemStack();
            if (stack.getAmount() <= 1) {
                tasteItem.remove();
            } else {
                stack.setAmount(stack.getAmount() - 1);
                tasteItem.setItemStack(stack);
            }
        }
    }

    private void startFlight(Location loc, BrainLink.Motor m) {
        Vector f = forward(loc.getYaw());
        Vector l = left(f);
        Vector dir = f.clone().multiply(m.fwd()).add(l.clone().multiply(m.left()));
        if (dir.lengthSquared() < 1e-6) dir = f.clone().multiply(-1);   // nel dubbio, indietro
        dir.normalize();

        vel = dir.multiply(TAKEOFF_SPEED).setY(TAKEOFF_LIFT);
        state = State.FLYING;
        flightTicks = 0;
        takeoffs++;
        loc.getWorld().spawnParticle(Particle.CLOUD, loc, 6, 0.15, 0.1, 0.15, 0.02);
    }

    private void flightStep(Location loc) {
        flightTicks++;

        Location next = loc.clone().add(vel);
        if (!free(next)) {                               // muro: solo in verticale
            vel.setX(0).setZ(0);
            next = loc.clone().add(vel);
            if (!free(next)) {                           // soffitto o pavimento
                vel.setY(0);
                next = loc.clone();
            }
        }

        double hs = Math.hypot(vel.getX(), vel.getZ());
        float yaw = hs > 0.05
                ? (float) Math.toDegrees(Math.atan2(-vel.getX(), vel.getZ()))
                : loc.getYaw();
        next.setYaw(yaw);
        next.setPitch(0);
        fly.teleport(next);

        vel.multiply(DRAG);
        vel.setY(vel.getY() - SINK);

        if (flightTicks > MAX_FLIGHT_TICKS || (flightTicks > 10 && vel.length() < 0.05)) {
            state = State.GROUND;                        // atterra: riprende a camminare
            baseY = next.getY();
            walkFwd = walkTurn = 0;
        }
    }

    // ------------------------------------------------------------- utilità

    /** Altezza a cui stare: poco sopra il primo blocco solido sotto di sé. */
    private double groundTarget(Location loc) {
        for (int i = 0; i <= GROUND_SCAN; i++) {
            Block b = loc.clone().subtract(0, i, 0).getBlock();
            if (!b.isPassable()) return b.getY() + 1 + cfg.hoverHeight();
        }
        return loc.getY() - 0.5;                         // niente sotto: scende piano
    }

    private static boolean free(Location loc) {
        return loc.getBlock().isPassable() && loc.clone().add(0, 0.6, 0).getBlock().isPassable();
    }

    /** Angolo di un vettore rispetto alla mosca, in gradi, positivo = a sinistra. */
    private static double angleOf(Vector d, Vector f, Vector l) {
        return Math.toDegrees(Math.atan2(d.dot(l), d.dot(f)));
    }

    /** Direzione in cui guarda la mosca (yaw di Minecraft: 0 = sud, 90 = ovest). */
    private static Vector forward(float yaw) {
        double r = Math.toRadians(yaw);
        return new Vector(-Math.sin(r), 0, Math.cos(r));
    }

    /** Sinistra della mosca, vista dall'alto. */
    private static Vector left(Vector f) {
        return new Vector(f.getZ(), 0, -f.getX());
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static double round(double v) {
        return Math.round(v * 100.0) / 100.0;
    }
}
