package it.project.fly;

import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.format.NamedTextColor;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.Particle;
import org.bukkit.Sound;
import org.bukkit.World;
import org.bukkit.block.Block;
import org.bukkit.block.data.BlockData;
import org.bukkit.block.data.Levelled;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.block.Action;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.inventory.EquipmentSlot;
import org.bukkit.inventory.ItemStack;
import org.bukkit.scheduler.BukkitTask;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * The nest: a composter filled ONLY with fruit, instead of producing bone meal,
 * hatches a fly (like Drosophila, which lays its eggs in fermenting fruit).
 *
 *  - Each composter has a "fruit only" state that starts true when it is empty.
 *  - A player who puts in a non-fruit item (or a hopper filling it, which we see
 *    as a level changed by someone else) turns it into a normal composter
 *    until it is emptied.
 *  - When fruit fills it (level 7), the composter becomes a nest: for
 *    nest.hatch-seconds seconds it "ferments" (particles, locked), then the fly is born,
 *    owned by whoever put in the last fruit.
 *  - If the fly cannot be born (the player already has theirs, or the brain PC is at
 *    its limit) the nest waits and retries every 5 seconds.
 *
 * In a vanilla composter level 7 turns into 8 (bone meal ready) after one second:
 * while hatching we keep the level at 6, so the bone meal never appears.
 */
public final class NestListener implements Listener {

    private static final int HATCH_LEVEL = 6;     // level kept while hatching
    private static final int FULL = 7;
    private static final int RETRY_TICKS = 100;   // 5 s

    private record Key(UUID world, int x, int y, int z) {
        static Key of(Block b) {
            return new Key(b.getWorld().getUID(), b.getX(), b.getY(), b.getZ());
        }
    }

    private static final class Nest {
        int level;                 // last level we saw
        boolean pure = true;       // only fruit so far
        UUID owner;                // who put in the last fruit
        BukkitTask hatch;          // non-null = fermenting
        int ticksLeft;
        boolean told;              // the owner already knows why it is not hatching
    }

    private final FlyManager flies;
    private final Settings cfg;
    private final Lang lang;
    private final Map<Key, Nest> nests = new HashMap<>();

    public NestListener(FlyManager flies) {
        this.flies = flies;
        this.cfg = flies.cfg;
        this.lang = flies.lang;
    }

    // ------------------------------------------------------------ clicks on the composter

    @EventHandler(priority = EventPriority.HIGH, ignoreCancelled = true)
    public void onInteract(PlayerInteractEvent e) {
        if (e.getAction() != Action.RIGHT_CLICK_BLOCK) return;
        Block b = e.getClickedBlock();
        if (b == null || b.getType() != Material.COMPOSTER) return;
        Player p = e.getPlayer();

        // the main hand takes precedence: the off hand only counts if the main hand is empty
        if (e.getHand() == EquipmentSlot.OFF_HAND && !p.getInventory().getItemInMainHand().getType().isAir()) return;
        if (e.getHand() != EquipmentSlot.HAND && e.getHand() != EquipmentSlot.OFF_HAND) return;

        Key key = Key.of(b);
        Nest n = nests.get(key);
        if (n != null && n.hatch != null) {
            e.setCancelled(true);
            p.sendActionBar(Component.text(lang.get("nest.stirring"), NamedTextColor.GREEN));
            return;
        }

        int before = level(b);
        if (before < 0) return;
        ItemStack item = e.getItem();
        boolean fruit = item != null && cfg.nestFruits().contains(item.getType());
        UUID who = p.getUniqueId();

        // we see the result (fruit accepted or not) on the next tick, once vanilla handling is done
        Bukkit.getScheduler().runTaskLater(flies.plugin, () -> after(b, before, fruit, who), 1L);
    }

    private void after(Block b, int before, boolean fruit, UUID who) {
        int now = level(b);
        if (now < 0) return;
        Key key = Key.of(b);
        Nest n = nests.computeIfAbsent(key, k -> {
            Nest fresh = new Nest();
            fresh.level = before;
            fresh.pure = before == 0;      // never seen before: it is "fruit only" only if it was empty
            return fresh;
        });
        if (n.hatch != null) return;

        // someone else changed it (hopper, emptied...) since we last saw it
        if (n.level != before) n.pure = before == 0;
        if (before == 0) {
            n.pure = true;
            n.owner = null;
        }

        if (fruit) {
            n.owner = who;
            if (n.pure && now > before && before == 0) {
                Player p = Bukkit.getPlayer(who);
                if (p != null) p.sendActionBar(Component.text(lang.get("nest.fruit_scent"), NamedTextColor.GREEN));
            }
        } else if (now != before && now != 0) {
            // something that is not fruit went in
            if (n.pure && before > 0) {
                Player p = Bukkit.getPlayer(who);
                if (p != null) p.sendActionBar(Component.text(
                        lang.get("nest.not_pure"), NamedTextColor.GRAY));
            }
            n.pure = false;
        }
        n.level = now;

        if (fruit && n.pure && now >= FULL) startHatch(b, n);
    }

    // ------------------------------------------------------------ hatching

    private void startHatch(Block b, Nest n) {
        setLevel(b, HATCH_LEVEL);
        n.level = HATCH_LEVEL;
        n.ticksLeft = Math.max(1, cfg.hatchSeconds()) * 20;
        n.told = false;
        Location center = b.getLocation().add(0.5, 1.0, 0.5);
        b.getWorld().playSound(center, Sound.BLOCK_COMPOSTER_FILL_SUCCESS, 1f, 0.6f);
        Player owner = n.owner != null ? Bukkit.getPlayer(n.owner) : null;
        if (owner != null) {
            owner.sendActionBar(Component.text(lang.get("nest.fermenting"),
                    NamedTextColor.GREEN));
        }
        n.hatch = Bukkit.getScheduler().runTaskTimer(flies.plugin, () -> hatchTick(b, n), 20L, 20L);
    }

    private void hatchTick(Block b, Nest n) {
        if (!b.getWorld().isChunkLoaded(b.getX() >> 4, b.getZ() >> 4)) return;   // nobody nearby: paused
        if (b.getType() != Material.COMPOSTER) {          // broken or replaced: no fly
            stop(b, n);
            return;
        }
        if (level(b) != HATCH_LEVEL) setLevel(b, HATCH_LEVEL);   // a hopper must not fill it
        World w = b.getWorld();
        Location center = b.getLocation().add(0.5, 1.0, 0.5);
        w.spawnParticle(Particle.COMPOSTER, center, 4, 0.25, 0.1, 0.25, 0);

        n.ticksLeft -= 20;
        if (n.ticksLeft > 0) return;

        FlyManager.Refusal no = flies.refusal(n.owner);
        if (no != null) {
            // wait and retry; tell the owner once (the server's 3 s wait doesn't count)
            if (!n.told && no != FlyManager.Refusal.WAIT) {
                Player p = n.owner != null ? Bukkit.getPlayer(n.owner) : null;
                if (p != null) {
                    p.sendMessage(Component.text(lang.get("nest.blocked",
                            "reason", flies.describe(no, n.owner)), NamedTextColor.YELLOW));
                }
                n.told = true;
            }
            n.ticksLeft = RETRY_TICKS;
            return;
        }

        Location at = center.clone();
        at.setYaw((float) (Math.random() * 360.0));
        FlyController fly = flies.spawn(at, n.owner, false);
        if (fly == null) {                                // refused at the last moment: retry
            n.ticksLeft = RETRY_TICKS;
            return;
        }
        setLevel(b, 0);
        w.playSound(center, Sound.BLOCK_COMPOSTER_EMPTY, 1f, 1.4f);
        w.spawnParticle(Particle.HAPPY_VILLAGER, center, 12, 0.3, 0.3, 0.3, 0);
        Player p = n.owner != null ? Bukkit.getPlayer(n.owner) : null;
        if (p != null) {
            p.sendMessage(Component.text(lang.get("nest.hatched", "id", fly.id()),
                    NamedTextColor.GREEN));
        }
        stop(b, n);
    }

    private void stop(Block b, Nest n) {
        if (n.hatch != null) n.hatch.cancel();
        n.hatch = null;
        nests.remove(Key.of(b));
    }

    /** Stops all hatchings (plugin shutdown). The composters stay at level 6. */
    public void shutdown() {
        for (Nest n : nests.values()) if (n.hatch != null) n.hatch.cancel();
        nests.clear();
    }

    // ------------------------------------------------------------ levels

    private static int level(Block b) {
        if (b.getType() != Material.COMPOSTER) return -1;
        BlockData d = b.getBlockData();
        return d instanceof Levelled l ? l.getLevel() : -1;
    }

    private static void setLevel(Block b, int level) {
        BlockData d = b.getBlockData();
        if (d instanceof Levelled l) {
            l.setLevel(level);
            b.setBlockData(l);
        }
    }
}
