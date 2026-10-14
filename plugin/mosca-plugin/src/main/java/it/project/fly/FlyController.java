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
import org.bukkit.block.BlockFace;
import org.bukkit.block.data.Waterlogged;
import org.bukkit.entity.Ageable;
import org.bukkit.entity.Breedable;
import org.bukkit.entity.Mob;
import org.bukkit.entity.Item;
import org.bukkit.entity.Player;
import org.bukkit.inventory.ItemStack;
import org.bukkit.persistence.PersistentDataType;
import org.bukkit.util.Vector;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.UUID;

/**
 * Una mosca (un Allay, un'ape... vedi body in config.yml) con il suo cervello. A ogni tick il FlyManager:
 *   1. le chiede i sensi (player che si avvicinano, odori, gusto, tatto): senses()
 *   2. dopo averli mandati al server, le fa applicare i comandi del suo cervello: act()
 * Con l'AI spenta il corpo ignora la velocità, quindi lo spostiamo noi con un teleport.
 */
public final class FlyController {

    /** GROUND: cammina; FLYING: salto di fuga (balistico); CRUISE: volo libero, guidato dal midollo. */
    public enum State { GROUND, FLYING, CRUISE }

    /** Perché sta volando: esplora, segue un player che le piace, va verso un odore. */
    private enum Goal { WANDER, FOLLOW, ODOR }

    // Fisica del volo di fuga (blocchi per tick)
    private static final double TAKEOFF_SPEED = 0.6;
    private static final double TAKEOFF_LIFT = 0.35;
    private static final double DRAG = 0.9;
    private static final double SINK = 0.025;
    private static final int MAX_FLIGHT_TICKS = 40;
    private static final int GROUND_SCAN = 4;        // quanti blocchi sotto cerca il terreno
    private static final int WATER_SCAN = 64;        // quanti blocchi sopra cerca il pelo dell'acqua
    private static final Set<Material> WATER_PLANTS = EnumSet.of(
            Material.KELP, Material.KELP_PLANT, Material.SEAGRASS, Material.TALL_SEAGRASS, Material.BUBBLE_COLUMN);
    private static final double TASTE_DIST = 1.0;    // blocchi: "ci sono sopra", può assaggiare
    private static final int BITE_TICKS = 60;        // ogni 3 s di pasto consuma un item
    private static final double RECOGNIZE = 3.5;     // blocchi: fin qui il corpo fungiforme riconosce un player
    private static final double LIKING_DECAY = Math.exp(-1.0 / (20.0 * 1800.0));   // come la memoria: ~mezz'ora

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

    private Mob fly;
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
    private Goal goal = Goal.WANDER;
    private String goalOdor;                         // ODOR: l'odore verso cui vola
    private float heading;                           // CRUISE: direzione di volo (yaw)
    private float saccadeLeft;                       // gradi di virata rapida ancora da fare
    private int saccadeIn;                           // tick al prossimo saccade
    private int cruiseTicks;                         // quanto vuole ancora volare
    private int calmTicks;                           // FOLLOW: da quanto è vicina al player
    private final Random rng = new Random();
    // giudizio della mosca su ogni player (la valenza letta quando lo riconosceva da solo e da
    // vicino); svanisce come la memoria. Serve solo a sapere da chi teletrasportarsi.
    private final Map<UUID, Double> liking = new HashMap<>();
    private UUID bonded;                             // a chi è legata adesso (cambia solo con un margine)
    private int teleportCooldown;
    private double pain;                             // dolore di un colpo (0..1): la punizione per la memoria
    private double memValence;                       // cosa ricorda dell'odore di player che sente (smussato)
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
        fly = loc.getWorld().spawn(loc, cfg.bodyClass(), a -> {
            a.setAI(false);
            a.setGravity(false);
            a.setCanPickupItems(false);
            a.setPersistent(false);          // non viene salvata: al riavvio sparisce
            a.setRemoveWhenFarAway(false);
            a.setSilent(true);
            a.customName(Component.text(cfg.bodyName() + " " + id));   // i cuori arrivano con refreshName()
            a.setCustomNameVisible(true);
            a.getPersistentDataContainer().set(MoscaPlugin.FLY_KEY, PersistentDataType.BYTE, (byte) 1);
            if (cfg.bodyBaby() && a instanceof Ageable young) {          // cucciolo (ape, pappagallo...)
                young.setBaby();
                if (a instanceof Breedable b) b.setAgeLock(true);         // e resta cucciolo
            }
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
                p.sendMessage(Component.text(cfg.bodyName() + " " + id + " è morta.", NamedTextColor.GRAY));
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
            fly.customName(Component.text(cfg.bodyName() + " " + id + " ")
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
    public String mode() {
        if (state == State.FLYING) return "in fuga";
        if (state == State.CRUISE) {
            return switch (goal) {
                case FOLLOW -> "vola dietro a un player che conosce";
                case ODOR -> "vola verso " + goalOdor;
                default -> "vola";
            };
        }
        return mode;
    }
    public double memory() { return memValence; }

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
        BrainLink.Motor last = brain.latest(id);
        if (last != null) {
            if (last.memFamiliar() >= 0.05) {             // il corpo fungiforme riconosce un odore (fino a ~4 blocchi)
                memValence += (last.memValence() - memValence) * 0.05;   // ti sente bene: segue la memoria (~1 s)
            } else {
                memValence *= 0.995;     // odore debole o sparito: l'attrazione resta per una decina di secondi
            }
        }
        updateLiking(last);

        BrainLink.Motor takeoff = brain.pollTakeoff(id);   // consumato anche durante il salto di fuga
        tookOff = reflexTakeoff;
        reflexTakeoff = false;
        if (state == State.FLYING) {
            flightStep(loc);
        } else if (takeoff != null) {                    // il Giant Fiber comanda, a terra e in volo
            startFlight(loc, takeoff);
            tookOff = true;
        } else if (state == State.CRUISE) {
            cruise(loc);
        } else {
            walk(loc);
        }
        if (tick % 20 == 0) maybeTeleport();
    }

    void writeLog(DebugLog log) {
        if (fly == null || fly.isDead()) return;
        BrainLink.Motor m = brain.latest(id);
        Location at = fly.getLocation();
        String base = String.format(Locale.ROOT, "%s,%d,%d,%s,%.2f,%.2f,%.2f,%.1f",
                id, log.elapsed(), tick, mode().replace(',', ' '), at.getX(), at.getY(), at.getZ(), at.getYaw());
        String tastePart = String.format(Locale.ROOT, ",%.2f,%.1f,%d,%.2f,%.2f,%.1f,%d,%.1f,%.2f,%.3f,%.3f,%.3f",
                taste, m == null ? 0.0 : m.mn9(), m != null && m.feeding() ? 1 : 0,
                touchL, touchR, m == null ? 0.0 : m.groom(), m != null && m.grooming() ? 1 : 0,
                fly.getHealth(), pain, m == null ? 0.0 : m.memValence(), memValence,
                m == null ? 0.0 : m.memFamiliar()) + String.format(Locale.ROOT, ",%.3f,%s",
                m == null ? 1.0 : m.loomGain(), bonded == null ? "" : bonded.toString());
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
                        "Mosca %s | tu: %s | vita %.0f | memoria %+.2f | looming %.2f/%.2f | GF %d/%d | MN9 %.0f | %s | spike %d | rt %.2f",
                        id, you, health(), memValence, m.loomL(), m.loomR(), m.gfL(), m.gfR(), m.mn9(), mode(),
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

            // antenne: l'odore del player stesso (ognuno ha il suo, lo sceglie il server)
            JsonObject body = new JsonObject();
            body.addProperty("odor", "player");
            body.addProperty("player", p.getUniqueId().toString());
            body.addProperty("angle", round(angle));
            body.addProperty("dist", round(dist));
            odors.add(body);
            sources.add(new Spinal.Source("player", angle, dist));
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
        msg.addProperty("pain", round(pain));
        pain *= 0.95;                                // il dolore dura di più: circa 1,5 s
        if (pain < 0.02) pain = 0;
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
        Spinal.Drive d = spinal.step(m != null ? m.perceive() : null, sources, blocked, memValence);
        mode = d.mode();

        // decollo: verso un player che le piace rimasto indietro, verso un odore attraente
        // lontano, oppure ogni tanto da sola mentre gironzola (le mosche vere volano spesso)
        Spinal.Source you = nearestPlayer();
        if (you != null && memValence >= cfg.followValence() && you.dist() > cfg.followDistance()) {
            startCruise(loc, Goal.FOLLOW, null, 200);
            return;
        }
        String far = farAttractiveOdor(m);
        if (far != null) {
            startCruise(loc, Goal.ODOR, far, 200);
            return;
        }
        boolean idle = "passeggia".equals(d.mode()) || "ferma".equals(d.mode());
        if (idle && rng.nextDouble() < cfg.flightChance()) {
            startCruise(loc, Goal.WANDER, null, 60 + rng.nextInt(100));
            return;
        }
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
        pain = Math.max(pain, strength);                   // dura circa un secondo e mezzo
        Vector f = forward(loc.getYaw());

        if (cfg.hitReflex() > 0 && strength >= cfg.hitReflex() && state != State.FLYING) {
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

    // ------------------------------------------------------------- volo libero
    //
    // Come per la camminata, nella mosca vera il ritmo del volo lo genera il cordone nervoso,
    // che FlyWire non contiene: lo fa il "midollo". Una Drosophila in volo fa tratti dritti
    // interrotti da saccadi (virate rapide di 30-120 gradi) e verso un odore attraente vira e
    // vola dritta. Il cervello resta al comando della fuga: il Giant Fiber la fa scappare
    // anche in aria.

    private void startCruise(Location loc, Goal g, String odor, int ticks) {
        state = State.CRUISE;
        goal = g;
        goalOdor = odor;
        cruiseTicks = ticks;
        heading = loc.getYaw();
        saccadeLeft = 0;
        saccadeIn = 10 + rng.nextInt(20);
        calmTicks = 0;
        walkFwd = walkTurn = 0;
    }

    private void cruise(Location loc) {
        cruiseTicks--;
        Spinal.Source target = goal == Goal.FOLLOW ? nearestPlayer()
                : goal == Goal.ODOR ? nearestSource(goalOdor) : null;
        if (goal == Goal.FOLLOW && memValence < cfg.memoryMin()) {   // non ti riconosce più: esplora un po'
            goal = Goal.WANDER;
            target = null;
            cruiseTicks = Math.min(cruiseTicks, 40 + rng.nextInt(40));
        }
        double speed = cfg.flightSpeed() / 20.0;
        if (target != null) {
            heading -= (float) clamp(target.angle(), -25, 25);      // virata rapida verso la sorgente
            double stop = goal == Goal.FOLLOW ? cfg.keepDistance() + 0.5 : 1.6;
            if (target.dist() < stop) {
                speed *= 0.2;                                         // arrivata: rallenta e si posa
                if (goal == Goal.ODOR || ++calmTicks > 40) {
                    land(loc);
                    return;
                }
            } else {
                calmTicks = 0;
                if (target.dist() > cfg.followDistance() * 2) speed *= 1.5;   // lontano: accelera
            }
            cruiseTicks = Math.max(cruiseTicks, 20);                  // finché ha la sorgente continua
        } else {
            if (goal != Goal.WANDER && cruiseTicks > 60) cruiseTicks = 60;     // persa: la cerca ancora un po'
            if (saccadeLeft != 0) {                                   // saccade in corso
                float step = (float) clamp(saccadeLeft, -35, 35);
                heading += step;
                saccadeLeft -= step;
            } else if (--saccadeIn <= 0) {
                saccadeLeft = (float) ((rng.nextBoolean() ? 1 : -1) * (30 + rng.nextDouble() * 90));
                saccadeIn = 10 + rng.nextInt(25);
            }
        }
        if (cruiseTicks <= 0) {
            land(loc);
            return;
        }

        Location next = loc.clone().add(forward(heading).multiply(speed));
        double ground = groundTarget(next) - cfg.hoverHeight();
        // quota: in crociera flight.height; quando arriva sulla frutta scende per posarsi
        boolean landing = goal == Goal.ODOR && target != null && target.dist() < 4.0;
        double wantY = ground + (landing ? cfg.hoverHeight() : cfg.flightHeight());
        next.setY(loc.getY() + clamp((wantY - loc.getY()) * 0.15, -0.15, 0.15) + 0.03 * Math.sin(tick * 0.6));
        if (!free(next)) {
            Location up = next.clone().add(0, 0.4, 0);
            if (free(up)) {
                next = up;                                            // ostacolo basso: sale
            } else {                                                  // muro: saccade per evitarlo
                saccadeLeft = (float) ((rng.nextBoolean() ? 1 : -1) * (100 + rng.nextDouble() * 60));
                next = loc.clone();
            }
        }
        next.setYaw(heading);
        next.setPitch(0);
        fly.teleport(next);
    }

    private void land(Location loc) {
        state = State.GROUND;
        goal = Goal.WANDER;
        baseY = loc.getY();                    // la camminata la riporta piano all'altezza del terreno
        walkFwd = walkTurn = 0;
    }

    /** Un odore attraente che il cervello percepisce, ma lontano: ci vola invece di camminare. */
    private String farAttractiveOdor(BrainLink.Motor m) {
        if (m == null) return null;
        for (Map.Entry<String, Double> e : m.perceive().entrySet()) {
            if (cfg.valence().getOrDefault(e.getKey(), 0.0) <= 0 || e.getValue() < 0.1) continue;
            Spinal.Source src = nearestSource(e.getKey());
            if (src != null && src.dist() > cfg.flyToOdorDistance()) return e.getKey();
        }
        return null;
    }

    private Spinal.Source nearestSource(String odor) {
        Spinal.Source best = null;
        for (Spinal.Source s : sources) {
            if (s.odor().equals(odor) && (best == null || s.dist() < best.dist())) best = s;
        }
        return best;
    }

    // ------------------------------------------------------------- legame e teletrasporto

    /** Aggiorna il giudizio sul player che la mosca sta riconoscendo, se è uno solo vicino. */
    private void updateLiking(BrainLink.Motor last) {
        liking.replaceAll((k, v) -> v * LIKING_DECAY);
        if (last == null || last.memFamiliar() < 0.05) return;
        UUID only = null;
        int near = 0;
        for (Map.Entry<UUID, double[]> e : lastSeen.entrySet()) {
            if (e.getValue()[0] <= RECOGNIZE) {
                near++;
                only = e.getKey();
            }
        }
        if (near == 1) {
            double old = liking.getOrDefault(only, 0.0);
            liking.put(only, old + (last.memValence() - old) * 0.05);
        }
    }

    /**
     * Il player a cui è legata: quello che le piace di più (almeno teleport-valence). Cambia
     * solo se un altro le piace nettamente di più (+0.1), sennò salterebbe avanti e indietro
     * tra due player che le piacciono quasi uguale.
     */
    public UUID favorite() {
        double cur = bonded != null ? liking.getOrDefault(bonded, 0.0) : 0.0;
        if (bonded != null && cur < cfg.teleportValence()) {
            bonded = null;
            cur = 0.0;
        }
        UUID best = bonded;
        double need = bonded != null ? cur + 0.1 : cfg.teleportValence();
        for (Map.Entry<UUID, Double> e : liking.entrySet()) {
            if (!e.getKey().equals(bonded) && e.getValue() >= need) {
                best = e.getKey();
                need = e.getValue();
            }
        }
        bonded = best;
        return bonded;
    }

    public double likingOf(UUID player) {
        return liking.getOrDefault(player, 0.0);
    }

    /**
     * Come gli animali domestici di Minecraft: se il player che le piace di più è troppo lontano
     * (o in un altro mondo) gli compare accanto. Una mosca vera non lo farebbe: si spegne con
     * memory.teleport-distance: 0.
     */
    private void maybeTeleport() {
        if (teleportCooldown > 0) teleportCooldown -= 20;
        if (cfg.teleportDistance() <= 0 || state == State.FLYING || teleportCooldown > 0) return;
        if (taste > 0 && feedTicks > 0) return;                       // sta mangiando
        UUID fav = favorite();
        Player p = fav != null ? Bukkit.getPlayer(fav) : null;
        if (p == null || p.isDead() || p.getGameMode() == GameMode.SPECTATOR) return;
        Location at = fly.getLocation();
        double td2 = cfg.teleportDistance() * cfg.teleportDistance();
        for (Player q : at.getWorld().getPlayers()) {                 // è già con qualcuno che le piace: resta
            if (q.getGameMode() != GameMode.SPECTATOR && liking.getOrDefault(q.getUniqueId(), 0.0) >= cfg.teleportValence()
                    && q.getLocation().distanceSquared(at) <= td2) return;
        }
        boolean far = p.getWorld() != at.getWorld()
                || p.getLocation().distanceSquared(at) > cfg.teleportDistance() * cfg.teleportDistance();
        if (!far) return;
        if (teleportTo(p)) teleportCooldown = 200;                    // al massimo uno ogni 10 s
    }

    /** La porta accanto al player (anche da un altro mondo). False se lì non c'è posto. */
    public boolean teleportTo(Player p) {
        if (fly == null || fly.isDead()) return false;
        Location dest = landingNear(p);
        if (dest == null) return false;
        Location at = fly.getLocation();
        at.getWorld().spawnParticle(Particle.PORTAL, at.clone().add(0, 0.5, 0), 24, 0.2, 0.3, 0.2, 0.4);
        fly.teleport(dest);
        dest.getWorld().spawnParticle(Particle.PORTAL, dest.clone().add(0, 0.5, 0), 24, 0.2, 0.3, 0.2, 0.4);
        vel = new Vector();
        land(dest);
        lastPos.clear();                                              // niente looming finto dal salto
        return true;
    }

    /** Un posto libero vicino al player, dietro di lui, rivolto verso di lui. */
    private static Location landingNear(Player p) {
        Location base = p.getLocation();
        Vector back = base.getDirection().setY(0);
        if (back.lengthSquared() < 1e-6) back = new Vector(0, 0, 1);
        back.normalize().multiply(-2);
        for (double rot : new double[]{0, Math.PI / 2, -Math.PI / 2, Math.PI}) {
            Location c = base.clone().add(back.clone().rotateAroundY(rot)).add(0, 1.0, 0);
            if (free(c)) {
                Vector look = base.toVector().subtract(c.toVector());
                c.setYaw((float) Math.toDegrees(Math.atan2(-look.getX(), look.getZ())));
                c.setPitch(0);
                return c;
            }
        }
        return null;
    }

    private Spinal.Source nearestPlayer() {
        Spinal.Source best = null;
        for (Spinal.Source s : sources) {
            if ("player".equals(s.odor()) && (best == null || s.dist() < best.dist())) best = s;
        }
        return best;
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
        // l'acqua la ferma come un pavimento: si posa sul pelo dell'acqua
        Block nb = next.getBlock();
        if (isWater(nb)) {
            double surf = waterSurface(nb);
            if (!Double.isNaN(surf)) {
                next.setY(surf + cfg.hoverHeight());
                if (vel.getY() < 0) vel.setY(0);
            }
        }
        fly.teleport(next);

        vel.multiply(DRAG);
        vel.setY(vel.getY() - SINK);

        if (flightTicks > MAX_FLIGHT_TICKS || (flightTicks > 10 && vel.length() < 0.05)) {
            double above = next.getY() - (groundTarget(next) - cfg.hoverHeight());
            if (above > 1.2) {
                startCruise(next, Goal.WANDER, null, 40 + rng.nextInt(40));   // in aria: continua a volare via
            } else {
                state = State.GROUND;                    // atterra: riprende a camminare
                baseY = next.getY();
                walkFwd = walkTurn = 0;
            }
        }
    }

    // ------------------------------------------------------------- utilità

    /**
     * Altezza a cui stare: poco sopra il primo blocco solido sotto di sé. L'acqua conta
     * come pavimento (la mosca sta sul pelo dell'acqua); se è già sotto, risale a galla.
     */
    private double groundTarget(Location loc) {
        Block here = loc.getBlock();
        if (isWater(here)) {
            double surf = waterSurface(here);
            return Double.isNaN(surf) ? loc.getY() : surf + cfg.hoverHeight();   // sotto un soffitto: resta
        }
        for (int i = 0; i <= GROUND_SCAN; i++) {
            Block b = loc.clone().subtract(0, i, 0).getBlock();
            if (isWater(b)) {
                double surf = waterSurface(b);
                if (!Double.isNaN(surf)) return surf + cfg.hoverHeight();
            }
            if (!b.isPassable()) return b.getY() + 1 + cfg.hoverHeight();
        }
        return loc.getY() - 0.5;                         // niente sotto: scende piano
    }

    /** Acqua (o lava), anche piante acquatiche e blocchi attraversabili pieni d'acqua. */
    private static boolean isWater(Block b) {
        if (b.isLiquid() || WATER_PLANTS.contains(b.getType())) return true;
        return b.isPassable() && b.getBlockData() instanceof Waterlogged w && w.isWaterlogged();
    }

    /** Quota del pelo dell'acqua sopra questo blocco d'acqua, o NaN se sopra c'è un soffitto. */
    private static double waterSurface(Block b) {
        for (int i = 0; i < WATER_SCAN; i++) {
            Block up = b.getRelative(BlockFace.UP);
            if (!isWater(up)) return up.isPassable() ? b.getY() + 1.0 : Double.NaN;
            b = up;
        }
        return Double.NaN;
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
