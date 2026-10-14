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
 * One fly (an Allay, a bee... see body in config.yml) with its brain. Every tick the FlyManager:
 *   1. asks it for its senses (approaching players, odors, taste, touch): senses()
 *   2. after sending them to the server, has it apply its brain's commands: act()
 * With the AI turned off the body ignores velocity, so we move it ourselves with a teleport.
 */
public final class FlyController {

    /** GROUND: walking; FLYING: escape jump (ballistic); CRUISE: free flight, driven by the spinal cord. */
    public enum State { GROUND, FLYING, CRUISE }

    /** Why it is flying: exploring, following a player it likes, heading for an odor. */
    private enum Goal { WANDER, FOLLOW, ODOR }

    // Escape-flight physics (blocks per tick)
    private static final double TAKEOFF_SPEED = 0.6;
    private static final double TAKEOFF_LIFT = 0.35;
    private static final double DRAG = 0.9;
    private static final double SINK = 0.025;
    private static final int MAX_FLIGHT_TICKS = 40;
    private static final int GROUND_SCAN = 4;        // how many blocks below to search for the ground
    private static final int WATER_SCAN = 64;        // how many blocks above to search for the water surface
    private static final Set<Material> WATER_PLANTS = EnumSet.of(
            Material.KELP, Material.KELP_PLANT, Material.SEAGRASS, Material.TALL_SEAGRASS, Material.BUBBLE_COLUMN);
    private static final double TASTE_DIST = 1.0;    // blocks: "standing on it", can taste
    private static final int BITE_TICKS = 60;        // every 3 s of feeding consumes one item
    private static final double RECOGNIZE = 3.5;     // blocks: up to here the mushroom body recognizes a player
    private static final double LIKING_DECAY = Math.exp(-1.0 / (20.0 * 1800.0));   // like the memory: ~half an hour

    private final String id;
    private final UUID owner;                        // player who spawned it (null = nobody)
    private final FlyManager mgr;
    private final JavaPlugin plugin;
    private final BrainLink brain;
    private final Settings cfg;
    private final Lang lang;
    private final Spinal spinal;
    private final List<Spinal.Source> sources = new ArrayList<>();
    private boolean blocked;
    private String mode = "idle";                    // internal English key, see Spinal.Drive
    private String modeOdor;                         // odor of following_odor / avoiding_odor, else null

    private Mob fly;
    private State state = State.GROUND;
    private Vector vel = new Vector();
    private int flightTicks;
    private double baseY;
    private double walkFwd, walkTurn;                // smoothed walking commands
    private long tick;
    private int takeoffs;
    private final Map<UUID, Vector> lastPos = new HashMap<>();     // last position of each player
    private final Map<UUID, double[]> lastSeen = new HashMap<>();   // debug: distance, speed, angle
    private final Map<UUID, String> heldOdor = new HashMap<>();     // debug: odor held in hand
    private boolean tookOff;
    private Goal goal = Goal.WANDER;
    private String goalOdor;                         // ODOR: the odor it is flying toward
    private float heading;                           // CRUISE: flight heading (yaw)
    private float saccadeLeft;                       // degrees of rapid turn still to do
    private int saccadeIn;                           // ticks until the next saccade
    private int cruiseTicks;                         // how long it still wants to fly
    private int calmTicks;                           // FOLLOW: how long it has been close to the player
    private final Random rng = new Random();
    // the fly's opinion of each player (the valence read while it recognized them alone and up
    // close); fades like the memory. Only used to know whom to teleport to.
    private final Map<UUID, Double> liking = new HashMap<>();
    private UUID bonded;                             // whom it is bonded to now (changes only with a margin)
    private int teleportCooldown;
    private double pain;                             // pain from a hit (0..1): the punishment for the memory
    private double memValence;                       // what it remembers about the player odor it smells (smoothed)
    private boolean reflexTakeoff;                   // takeoff from the body reflex (not from the brain)
    private double lastHitAngle;
    private double taste;                            // sugar under its feet (0..1)
    private Item tasteItem;                          // ground item it is tasting
    private int feedTicks;
    private double touchL, touchR;                   // hit on the head, per side (0..1), fades
    private int groomTicks;
    private float groomYaw;

    FlyController(String id, UUID owner, FlyManager mgr) {
        this.id = id;
        this.owner = owner;
        this.mgr = mgr;
        this.plugin = mgr.plugin;
        this.brain = mgr.brain;
        this.cfg = mgr.cfg;
        this.lang = mgr.lang;
        this.spinal = new Spinal(cfg);
    }

    public String id() { return id; }
    public UUID owner() { return owner; }
    public UUID entityId() { return fly != null ? fly.getUniqueId() : null; }
    public Location location() { return fly != null && !fly.isDead() ? fly.getLocation() : null; }
    public double health() { return fly != null ? fly.getHealth() : 0; }

    // --------------------------------------------------------- life cycle

    void spawn(Location loc) {
        fly = loc.getWorld().spawn(loc, cfg.bodyClass(), a -> {
            a.setAI(false);
            a.setGravity(false);
            a.setCanPickupItems(false);
            a.setPersistent(false);          // not saved: it disappears on restart
            a.setRemoveWhenFarAway(false);
            a.setSilent(true);
            a.customName(Component.text(cfg.bodyName() + " " + id));   // the hearts come with refreshName()
            a.setCustomNameVisible(true);
            a.getPersistentDataContainer().set(MoscaPlugin.FLY_KEY, PersistentDataType.BYTE, (byte) 1);
            if (cfg.bodyBaby() && a instanceof Ageable young) {          // baby (bee, parrot...)
                young.setBaby();
                if (a instanceof Breedable b) b.setAgeLock(true);         // and stays a baby
            }
        });
        baseY = loc.getY();
        state = State.GROUND;
        refreshName();
        walkFwd = walkTurn = 0;
        takeoffs = 0;
        lastPos.clear();
    }

    /** Removes the entity from the world (the manager notifies the server). */
    void despawn() {
        if (fly != null) {
            fly.remove();
            fly = null;
        }
        lastPos.clear();
    }

    /** Dead: notifies the brain (which shuts down) and leaves the animation to Minecraft. */
    public void onDeath() {
        if (fly == null) return;
        Location loc = fly.getLocation();
        for (Player p : loc.getWorld().getPlayers()) {
            if (p.getLocation().distance(loc) < 48) {
                p.sendMessage(Component.text(lang.get("fly.died", "name", cfg.bodyName(), "id", id), NamedTextColor.GRAY));
            }
        }
        fly = null;                                  // no remove(): the death animation stays
        lastPos.clear();
        mgr.died(this);                              // the manager notifies the server: the brain stops
    }

    /** Name with hearts: red for full ones, grey for lost ones. */
    public void refreshName() {
        if (fly == null) return;
        Bukkit.getScheduler().runTask(plugin, () -> {   // the damage is applied after the event
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
    /**
     * Current behaviour as an internal English key: idle, wandering, following_odor, avoiding_odor,
     * following_player, avoiding_player, eating, grooming, escaping, flying, flying_to_player,
     * flying_to_odor. Shown to players through Lang (mode.<key>), written as-is to the debug CSV.
     */
    public String mode() {
        if (state == State.FLYING) return "escaping";
        if (state == State.CRUISE) {
            return switch (goal) {
                case FOLLOW -> "flying_to_player";
                case ODOR -> "flying_to_odor";
                default -> "flying";
            };
        }
        return mode;
    }

    /** The odor the current mode refers to (following_odor, avoiding_odor, flying_to_odor), or null. */
    public String modeOdor() {
        if (state == State.FLYING) return null;
        if (state == State.CRUISE) return goal == Goal.ODOR ? goalOdor : null;
        return modeOdor;
    }

    /** The current mode, translated for players. */
    public String modeLabel() {
        String odor = modeOdor();
        return lang.get("mode." + mode(), "odor", odor != null ? odor : "");
    }
    public double memory() { return memValence; }

    // --------------------------------------------------------------- tick

    /** First half of the tick: what the fly senses (null if it is not in the world right now). */
    JsonObject senses() {
        if (fly == null || fly.isDead() || !fly.isValid()) return null;   // chunk not loaded: wait
        return collectSenses(fly.getLocation());
    }

    /** Second half of the tick: applies the brain's commands. */
    void act() {
        if (fly == null || fly.isDead() || !fly.isValid()) return;
        tick++;
        Location loc = fly.getLocation();
        BrainLink.Motor last = brain.latest(id);
        if (last != null) {
            if (last.memFamiliar() >= 0.05) {             // the mushroom body recognizes an odor (up to ~4 blocks)
                memValence += (last.memValence() - memValence) * 0.05;   // smells you clearly: tracks the memory (~1 s)
            } else {
                memValence *= 0.995;     // odor weak or gone: the attraction lasts about ten seconds
            }
        }
        updateLiking(last);

        BrainLink.Motor takeoff = brain.pollTakeoff(id);   // consumed even during the escape jump
        tookOff = reflexTakeoff;
        reflexTakeoff = false;
        if (state == State.FLYING) {
            flightStep(loc);
        } else if (takeoff != null) {                    // the Giant Fiber is in command, on the ground and in flight
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
        // mode column: the English key, plus ":<odor>" when the mode refers to an odor
        String csvMode = modeOdor() != null ? mode() + ":" + modeOdor() : mode();
        String base = String.format(Locale.ROOT, "%s,%d,%d,%s,%.2f,%.2f,%.2f,%.1f",
                id, log.elapsed(), tick, csvMode.replace(',', ' '), at.getX(), at.getY(), at.getZ(), at.getYaw());
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

    /** Debug action-bar line for a player (refers to this fly). */
    String debugLine(Player p) {
        BrainLink.Motor m = brain.latest(id);
        double[] seen = lastSeen.get(p.getUniqueId());
        String you = seen == null ? lang.get("debug.out_of_range")
                : lang.get("debug.you", "dist", f("%.1f", seen[0]), "speed", f("%+.1f", seen[1]));
        return m == null
                ? lang.get("debug.no_brain", "id", id, "you", you, "mode", modeLabel())
                : lang.get("debug.line", "id", id, "you", you, "health", f("%.0f", health()),
                        "memory", f("%+.2f", memValence), "loomL", f("%.2f", m.loomL()), "loomR", f("%.2f", m.loomR()),
                        "gfL", m.gfL(), "gfR", m.gfR(), "mn9", f("%.0f", m.mn9()), "mode", modeLabel(),
                        "spikes", m.spikes(), "rt", f("%.2f", m.rt()));
    }

    // ---------------------------------------------------------------- senses

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

            // eyes: how fast the PLAYER is moving toward the fly.
            // We only count the player's movement, not the fly's: when the fly is the one
            // moving toward you (for the apple) it must not scare itself.
            seen.add(p.getUniqueId());
            Vector pos = p.getLocation().toVector();
            Vector prev = lastPos.put(p.getUniqueId(), pos);
            double speed = 0.0;
            if (prev != null && dist > 1e-6) {
                Vector vel = pos.clone().subtract(prev).multiply(20.0);    // blocks per second
                Vector towardFly = d.clone().multiply(-1.0 / dist);        // from the player to the fly
                speed = clamp(vel.dot(towardFly), -20, 20);
            }
            lastSeen.put(p.getUniqueId(), new double[]{dist, speed, angle});
            JsonObject t = new JsonObject();
            t.addProperty("angle", round(angle));
            t.addProperty("dist", round(dist));
            t.addProperty("speed", round(speed));
            threats.add(t);

            // antennae: the odor of the item held in hand
            String held = addOdor(odors, p.getInventory().getItemInMainHand(), angle, dist);
            String off = addOdor(odors, p.getInventory().getItemInOffHand(), angle, dist);
            if (dist <= TASTE_DIST + 0.9 && ("fruity".equals(held) || "fruity".equals(off))) {
                taste = 1.0;                         // it is on the hand holding the fruit
            }
            heldOdor.put(p.getUniqueId(), held != null ? held : (off != null ? off : ""));

            // antennae: the odor of the player themself (each has their own, chosen by the server)
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

        // antennae: items dropped on the ground
        for (Item item : loc.getNearbyEntitiesByType(Item.class, range)) {
            Vector d = item.getLocation().toVector().subtract(here);
            String odor = addOdor(odors, item.getItemStack(), angleOf(d, f, l), d.length());
            if ("fruity".equals(odor) && d.length() <= TASTE_DIST + cfg.hoverHeight()) {
                taste = 1.0;                         // standing on it: the legs touch the sugar
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
        pain *= 0.95;                                // pain lasts longer: about 1.5 s
        if (pain < 0.02) pain = 0;
        touchL *= 0.8;                               // touch fades in about 0.3 s
        touchR *= 0.8;
        if (touchL < 0.02) touchL = 0;
        if (touchR < 0.02) touchR = 0;
        return msg;
    }

    /** Adds the item's odor, if it has one; returns the odor name or null. */
    private String addOdor(JsonArray odors, ItemStack stack, double angle, double dist) {
        if (stack == null || stack.getType() == Material.AIR) return null;
        String odor = cfg.odors().get(stack.getType());
        if (odor == null) return null;
        JsonObject o = new JsonObject();
        o.addProperty("odor", odor);
        o.addProperty("angle", round(angle));
        o.addProperty("dist", round(dist));
        odors.add(o);
        sources.add(new Spinal.Source(odor, angle, dist));    // for the spinal cord
        return odor;
    }

    // --------------------------------------------------------------- motion

    /** On the ground (hovering just above): the spinal cord generates walking, weighted by what the brain perceives. */
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
        if (taste <= 0) feedTicks = 0;               // the meal restarts only if it moves away
        Spinal.Drive d = spinal.step(m != null ? m.perceive() : null, sources, blocked, memValence);
        mode = d.mode();
        modeOdor = d.odor();

        // takeoff: toward a liked player left behind, toward a distant attractive odor,
        // or now and then on its own while wandering (real flies fly often)
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
        boolean idle = "wandering".equals(d.mode()) || "idle".equals(d.mode());
        if (idle && rng.nextDouble() < cfg.flightChance()) {
            startCruise(loc, Goal.WANDER, null, 60 + rng.nextInt(100));
            return;
        }
        double a = cfg.smoothing();
        walkFwd += (d.fwd() - walkFwd) * a;
        walkTurn += (d.turn() - walkTurn) * a;

        // positive steering = left = decreasing yaw
        float yaw = loc.getYaw() - (float) (walkTurn * cfg.maxTurn() / 20.0);
        Vector step = forward(yaw).multiply(walkFwd * cfg.maxSpeed() / 20.0);

        Location next = loc.clone().add(step);
        blocked = false;
        if (!free(next)) {
            Location up = next.clone().add(0, 1, 0);       // step: try to climb it
            if (free(up) && free(loc.clone().add(0, 1, 0))) {
                next = up;
                baseY += 1;
            } else {
                next = loc.clone();                         // wall: rotation only
                blocked = step.lengthSquared() > 1e-6;
            }
        }

        baseY += (groundTarget(next) - baseY) * 0.2;
        next.setY(baseY + 0.05 * Math.sin(tick * 0.15));
        next.setYaw(yaw);
        next.setPitch(0);
        fly.teleport(next);
    }

    /** Applies touch on the side the stimulus comes from. Returns the direction of the stimulus. */
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
     * Hit (punch, sword, arrow...). The blow becomes touch on the head bristles on the side
     * that was hit, stronger the harder the blow; above the threshold the body reflex also
     * fires and it jumps away. Minecraft applies the damage (red flash included).
     */
    public void hit(Location from, double damage) {
        if (fly == null || !fly.isValid()) return;
        Location loc = fly.getLocation();
        double strength = clamp(damage / 6.0, 0.3, 1.0);   // punch ~0.3, sword ~1
        Vector d = feel(from, strength);
        pain = Math.max(pain, strength);                   // lasts about a second and a half
        Vector f = forward(loc.getYaw());

        if (cfg.hitReflex() > 0 && strength >= cfg.hitReflex() && state != State.FLYING) {
            // reflex: away from the side that was hit
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

    // ------------------------------------------------------------- free flight
    //
    // As with walking, in a real fly the flight rhythm is generated by the nerve cord,
    // which FlyWire does not contain: the "spinal cord" does it. A flying Drosophila makes
    // straight stretches interrupted by saccades (rapid 30-120 degree turns), and toward an
    // attractive odor it turns and flies straight. The brain stays in charge of escape: the
    // Giant Fiber makes it flee in mid-air too.

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
        if (goal == Goal.FOLLOW && memValence < cfg.memoryMin()) {   // no longer recognizes you: explore for a while
            goal = Goal.WANDER;
            target = null;
            cruiseTicks = Math.min(cruiseTicks, 40 + rng.nextInt(40));
        }
        double speed = cfg.flightSpeed() / 20.0;
        if (target != null) {
            heading -= (float) clamp(target.angle(), -25, 25);      // rapid turn toward the source
            double stop = goal == Goal.FOLLOW ? cfg.keepDistance() + 0.5 : 1.6;
            if (target.dist() < stop) {
                speed *= 0.2;                                         // arrived: slow down and land
                if (goal == Goal.ODOR || ++calmTicks > 40) {
                    land(loc);
                    return;
                }
            } else {
                calmTicks = 0;
                if (target.dist() > cfg.followDistance() * 2) speed *= 1.5;   // far away: speed up
            }
            cruiseTicks = Math.max(cruiseTicks, 20);                  // keep going as long as it has the source
        } else {
            if (goal != Goal.WANDER && cruiseTicks > 60) cruiseTicks = 60;     // lost: keep looking for a while
            if (saccadeLeft != 0) {                                   // saccade in progress
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
        // altitude: flight.height while cruising; when it reaches the fruit it descends to land
        boolean landing = goal == Goal.ODOR && target != null && target.dist() < 4.0;
        double wantY = ground + (landing ? cfg.hoverHeight() : cfg.flightHeight());
        next.setY(loc.getY() + clamp((wantY - loc.getY()) * 0.15, -0.15, 0.15) + 0.03 * Math.sin(tick * 0.6));
        if (!free(next)) {
            Location up = next.clone().add(0, 0.4, 0);
            if (free(up)) {
                next = up;                                            // low obstacle: climb
            } else {                                                  // wall: saccade to avoid it
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
        baseY = loc.getY();                    // walking slowly brings it back to ground height
        walkFwd = walkTurn = 0;
    }

    /** An attractive odor the brain perceives, but far away: fly there instead of walking. */
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

    // ------------------------------------------------------------- bonding and teleport

    /** Updates the opinion of the player the fly is recognizing, if only one is nearby. */
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
     * The player it is bonded to: the one it likes most (at least teleport-valence). Changes
     * only if another one is clearly liked more (+0.1), otherwise it would jump back and forth
     * between two players it likes almost equally.
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
     * Like Minecraft pets: if the player it likes most is too far away (or in another world)
     * it appears next to them. A real fly would not do this: turn it off with
     * memory.teleport-distance: 0.
     */
    private void maybeTeleport() {
        if (teleportCooldown > 0) teleportCooldown -= 20;
        if (cfg.teleportDistance() <= 0 || state == State.FLYING || teleportCooldown > 0) return;
        if (taste > 0 && feedTicks > 0) return;                       // it is eating
        UUID fav = favorite();
        Player p = fav != null ? Bukkit.getPlayer(fav) : null;
        if (p == null || p.isDead() || p.getGameMode() == GameMode.SPECTATOR) return;
        Location at = fly.getLocation();
        double td2 = cfg.teleportDistance() * cfg.teleportDistance();
        for (Player q : at.getWorld().getPlayers()) {                 // already with someone it likes: stay
            if (q.getGameMode() != GameMode.SPECTATOR && liking.getOrDefault(q.getUniqueId(), 0.0) >= cfg.teleportValence()
                    && q.getLocation().distanceSquared(at) <= td2) return;
        }
        boolean far = p.getWorld() != at.getWorld()
                || p.getLocation().distanceSquared(at) > cfg.teleportDistance() * cfg.teleportDistance();
        if (!far) return;
        if (teleportTo(p)) teleportCooldown = 200;                    // at most one every 10 s
    }

    /** Brings it next to the player (even from another world). False if there is no room there. */
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
        lastPos.clear();                                              // no fake looming from the jump
        return true;
    }

    /** A free spot near the player, behind them, facing them. */
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

    /** Bristle descending neurons active: the fly stops and rubs itself. */
    private void groom(Location loc) {
        mode = "grooming";
        modeOdor = null;
        walkFwd = 0;
        walkTurn = 0;
        if (groomTicks++ == 0) groomYaw = loc.getYaw();
        if (groomTicks % 4 == 1) fly.swingMainHand();             // the "legs" rubbing
        if (groomTicks % 8 == 1) {
            loc.getWorld().spawnParticle(Particle.WHITE_ASH, loc.clone().add(0, 0.5, 0), 4, 0.15, 0.1, 0.15, 0);
        }
        Location next = loc.clone();
        baseY += (groundTarget(next) - baseY) * 0.2;
        next.setY(baseY + 0.02 * Math.sin(groomTicks * 2.1));
        // small head jerks: the front legs rubbing the touched spot
        next.setYaw(groomYaw + (float) (Math.sin(groomTicks * 1.7) * 5.0));
        fly.teleport(next);
    }

    /** MN9 active: proboscis out, the fly stays still and eats. */
    private void eat(Location loc) {
        mode = "eating";
        modeOdor = null;
        walkFwd = 0;
        walkTurn = 0;
        feedTicks++;
        Location next = loc.clone();
        baseY += (groundTarget(next) - baseY) * 0.2;
        next.setY(baseY + 0.02 * Math.sin(tick * 0.5));         // small movements while eating
        fly.teleport(next);
        if (feedTicks % 10 == 0) {
            loc.getWorld().spawnParticle(Particle.HEART, loc.clone().add(0, 0.6, 0), 1, 0.1, 0.1, 0.1, 0);
        }
        if (feedTicks % BITE_TICKS == 0 && cfg.healPerBite() > 0) {   // eating restores health
            double max = fly.getAttribute(Attribute.MAX_HEALTH) != null
                    ? fly.getAttribute(Attribute.MAX_HEALTH).getValue() : 20.0;
            if (fly.getHealth() < max) {
                fly.setHealth(Math.min(max, fly.getHealth() + cfg.healPerBite()));
                refreshName();
            }
        }
        if (tasteItem != null && feedTicks % BITE_TICKS == 0) {   // one bite at a time
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
        if (dir.lengthSquared() < 1e-6) dir = f.clone().multiply(-1);   // when in doubt, backwards
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
        if (!free(next)) {                               // wall: vertical only
            vel.setX(0).setZ(0);
            next = loc.clone().add(vel);
            if (!free(next)) {                           // ceiling or floor
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
        // water stops it like a floor: it settles on the water surface
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
                startCruise(next, Goal.WANDER, null, 40 + rng.nextInt(40));   // in the air: keep flying away
            } else {
                state = State.GROUND;                    // lands: resumes walking
                baseY = next.getY();
                walkFwd = walkTurn = 0;
            }
        }
    }

    // ------------------------------------------------------------- utilities

    /**
     * Height to hold: just above the first solid block below. Water counts as a floor
     * (the fly stays on the water surface); if it is already underwater, it floats back up.
     */
    private double groundTarget(Location loc) {
        Block here = loc.getBlock();
        if (isWater(here)) {
            double surf = waterSurface(here);
            return Double.isNaN(surf) ? loc.getY() : surf + cfg.hoverHeight();   // under a ceiling: stay
        }
        for (int i = 0; i <= GROUND_SCAN; i++) {
            Block b = loc.clone().subtract(0, i, 0).getBlock();
            if (isWater(b)) {
                double surf = waterSurface(b);
                if (!Double.isNaN(surf)) return surf + cfg.hoverHeight();
            }
            if (!b.isPassable()) return b.getY() + 1 + cfg.hoverHeight();
        }
        return loc.getY() - 0.5;                         // nothing below: descend slowly
    }

    /** Water (or lava), including water plants and passable waterlogged blocks. */
    private static boolean isWater(Block b) {
        if (b.isLiquid() || WATER_PLANTS.contains(b.getType())) return true;
        return b.isPassable() && b.getBlockData() instanceof Waterlogged w && w.isWaterlogged();
    }

    /** Height of the water surface above this water block, or NaN if there is a ceiling above. */
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

    /** Angle of a vector relative to the fly, in degrees, positive = left. */
    private static double angleOf(Vector d, Vector f, Vector l) {
        return Math.toDegrees(Math.atan2(d.dot(l), d.dot(f)));
    }

    /** Direction the fly is facing (Minecraft yaw: 0 = south, 90 = west). */
    private static Vector forward(float yaw) {
        double r = Math.toRadians(yaw);
        return new Vector(-Math.sin(r), 0, Math.cos(r));
    }

    /** The fly's left, seen from above. */
    private static Vector left(Vector f) {
        return new Vector(f.getZ(), 0, -f.getX());
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static double round(double v) {
        return Math.round(v * 100.0) / 100.0;
    }

    /** Formats a number the same way regardless of the server locale. */
    private static String f(String format, double v) {
        return String.format(Locale.ROOT, format, v);
    }
}
