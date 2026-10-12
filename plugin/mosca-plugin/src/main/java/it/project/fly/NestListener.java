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
 * Il nido: una compostiera riempita SOLO di frutta, invece della farina d'ossa,
 * fa nascere una mosca (come la Drosophila, che depone le uova nella frutta che fermenta).
 *
 *  - Ogni compostiera ha uno stato "solo frutta" che parte vero quando è vuota.
 *  - Un player che ci mette un item non di frutta (o una tramoggia che la riempie,
 *    che vediamo come livello cambiato da qualcun altro) la rende una compostiera normale
 *    fino a quando non viene svuotata.
 *  - Quando la frutta la riempie (livello 7), la compostiera diventa un nido: per
 *    nest.hatch-seconds secondi "fermenta" (particelle, bloccata), poi nasce la mosca,
 *    che appartiene a chi ha messo l'ultimo frutto.
 *  - Se la mosca non può nascere (il player ha già la sua, o il PC dei cervelli è al
 *    limite) il nido aspetta e riprova ogni 5 secondi.
 *
 * Nella compostiera vanilla il livello 7 diventa 8 (farina d'ossa pronta) dopo un secondo:
 * durante la schiusa teniamo il livello a 6, così la farina non compare mai.
 */
public final class NestListener implements Listener {

    private static final int HATCH_LEVEL = 6;     // livello tenuto durante la schiusa
    private static final int FULL = 7;
    private static final int RETRY_TICKS = 100;   // 5 s

    private record Key(UUID world, int x, int y, int z) {
        static Key of(Block b) {
            return new Key(b.getWorld().getUID(), b.getX(), b.getY(), b.getZ());
        }
    }

    private static final class Nest {
        int level;                 // ultimo livello visto da noi
        boolean pure = true;       // finora solo frutta
        UUID owner;                // chi ha messo l'ultimo frutto
        BukkitTask hatch;          // non null = sta fermentando
        int ticksLeft;
        boolean told;              // il proprietario sa già perché non nasce
    }

    private final FlyManager flies;
    private final Settings cfg;
    private final Map<Key, Nest> nests = new HashMap<>();

    public NestListener(FlyManager flies) {
        this.flies = flies;
        this.cfg = flies.cfg;
    }

    // ------------------------------------------------------------ click sulla compostiera

    @EventHandler(priority = EventPriority.HIGH, ignoreCancelled = true)
    public void onInteract(PlayerInteractEvent e) {
        if (e.getAction() != Action.RIGHT_CLICK_BLOCK) return;
        Block b = e.getClickedBlock();
        if (b == null || b.getType() != Material.COMPOSTER) return;
        Player p = e.getPlayer();

        // la mano principale ha la precedenza: la mano secondaria conta solo se la principale è vuota
        if (e.getHand() == EquipmentSlot.OFF_HAND && !p.getInventory().getItemInMainHand().getType().isAir()) return;
        if (e.getHand() != EquipmentSlot.HAND && e.getHand() != EquipmentSlot.OFF_HAND) return;

        Key key = Key.of(b);
        Nest n = nests.get(key);
        if (n != null && n.hatch != null) {
            e.setCancelled(true);
            p.sendActionBar(Component.text("Qualcosa si muove nella frutta...", NamedTextColor.GREEN));
            return;
        }

        int before = level(b);
        if (before < 0) return;
        ItemStack item = e.getItem();
        boolean fruit = item != null && cfg.nestFruits().contains(item.getType());
        UUID who = p.getUniqueId();

        // il risultato (frutto accettato o no) lo vediamo al tick dopo, a giro vanilla finito
        Bukkit.getScheduler().runTaskLater(flies.plugin, () -> after(b, before, fruit, who), 1L);
    }

    private void after(Block b, int before, boolean fruit, UUID who) {
        int now = level(b);
        if (now < 0) return;
        Key key = Key.of(b);
        Nest n = nests.computeIfAbsent(key, k -> {
            Nest fresh = new Nest();
            fresh.level = before;
            fresh.pure = before == 0;      // mai vista prima: è "solo frutta" solo se era vuota
            return fresh;
        });
        if (n.hatch != null) return;

        // qualcun altro l'ha cambiata (tramoggia, svuotata...) da quando l'abbiamo vista
        if (n.level != before) n.pure = before == 0;
        if (before == 0) {
            n.pure = true;
            n.owner = null;
        }

        if (fruit) {
            n.owner = who;
            if (n.pure && now > before && before == 0) {
                Player p = Bukkit.getPlayer(who);
                if (p != null) p.sendActionBar(Component.text("Profumo di frutta...", NamedTextColor.GREEN));
            }
        } else if (now != before && now != 0) {
            // è entrato qualcosa che non è frutta
            if (n.pure && before > 0) {
                Player p = Bukkit.getPlayer(who);
                if (p != null) p.sendActionBar(Component.text(
                        "Non è più solo frutta: niente mosca da questa compostiera", NamedTextColor.GRAY));
            }
            n.pure = false;
        }
        n.level = now;

        if (fruit && n.pure && now >= FULL) startHatch(b, n);
    }

    // ------------------------------------------------------------ schiusa

    private void startHatch(Block b, Nest n) {
        setLevel(b, HATCH_LEVEL);
        n.level = HATCH_LEVEL;
        n.ticksLeft = Math.max(1, cfg.hatchSeconds()) * 20;
        n.told = false;
        Location center = b.getLocation().add(0.5, 1.0, 0.5);
        b.getWorld().playSound(center, Sound.BLOCK_COMPOSTER_FILL_SUCCESS, 1f, 0.6f);
        Player owner = n.owner != null ? Bukkit.getPlayer(n.owner) : null;
        if (owner != null) {
            owner.sendActionBar(Component.text("La frutta fermenta: qualcosa ci ha deposto le uova...",
                    NamedTextColor.GREEN));
        }
        n.hatch = Bukkit.getScheduler().runTaskTimer(flies.plugin, () -> hatchTick(b, n), 20L, 20L);
    }

    private void hatchTick(Block b, Nest n) {
        if (!b.getWorld().isChunkLoaded(b.getX() >> 4, b.getZ() >> 4)) return;   // nessuno vicino: in pausa
        if (b.getType() != Material.COMPOSTER) {          // rotta o sostituita: niente mosca
            stop(b, n);
            return;
        }
        if (level(b) != HATCH_LEVEL) setLevel(b, HATCH_LEVEL);   // una tramoggia non la deve riempire
        World w = b.getWorld();
        Location center = b.getLocation().add(0.5, 1.0, 0.5);
        w.spawnParticle(Particle.COMPOSTER, center, 4, 0.25, 0.1, 0.25, 0);

        n.ticksLeft -= 20;
        if (n.ticksLeft > 0) return;

        FlyManager.Refusal no = flies.refusal(n.owner);
        if (no != null) {
            // aspetta e riprova; il proprietario lo avvisiamo una volta (l'attesa di 3 s del server non conta)
            if (!n.told && no != FlyManager.Refusal.WAIT) {
                Player p = n.owner != null ? Bukkit.getPlayer(n.owner) : null;
                if (p != null) {
                    p.sendMessage(Component.text("Nella tua compostiera c'è una mosca pronta a nascere, ma: "
                            + flies.describe(no, n.owner), NamedTextColor.YELLOW));
                }
                n.told = true;
            }
            n.ticksLeft = RETRY_TICKS;
            return;
        }

        Location at = center.clone();
        at.setYaw((float) (Math.random() * 360.0));
        FlyController fly = flies.spawn(at, n.owner, false);
        if (fly == null) {                                // rifiutata all'ultimo: riprova
            n.ticksLeft = RETRY_TICKS;
            return;
        }
        setLevel(b, 0);
        w.playSound(center, Sound.BLOCK_COMPOSTER_EMPTY, 1f, 1.4f);
        w.spawnParticle(Particle.HAPPY_VILLAGER, center, 12, 0.3, 0.3, 0.3, 0);
        Player p = n.owner != null ? Bukkit.getPlayer(n.owner) : null;
        if (p != null) {
            p.sendMessage(Component.text("È nata la mosca " + fly.id() + " dalla tua compostiera!",
                    NamedTextColor.GREEN));
        }
        stop(b, n);
    }

    private void stop(Block b, Nest n) {
        if (n.hatch != null) n.hatch.cancel();
        n.hatch = null;
        nests.remove(Key.of(b));
    }

    /** Ferma tutte le schiuse (spegnimento del plugin). Le compostiere restano a livello 6. */
    public void shutdown() {
        for (Nest n : nests.values()) if (n.hatch != null) n.hatch.cancel();
        nests.clear();
    }

    // ------------------------------------------------------------ livelli

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
