package it.project.fly;

import org.bukkit.event.EventHandler;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.entity.EntityPickupItemEvent;
import org.bukkit.event.player.PlayerInteractAtEntityEvent;
import org.bukkit.event.entity.EntityDeathEvent;
import org.bukkit.event.player.PlayerInteractEntityEvent;

/**
 * Spegne le interazioni vanilla dell'Allay (dargli item, duplicarlo, fargli raccogliere
 * cose). Un colpo le toglie vita e diventa tatto per il cervello; se muore, muore davvero.
 */
public final class FlyListener implements Listener {

    private final FlyManager flies;

    public FlyListener(FlyManager flies) {
        this.flies = flies;
    }

    @EventHandler(ignoreCancelled = true)
    public void onInteract(PlayerInteractEntityEvent e) {
        if (flies.isFly(e.getRightClicked())) e.setCancelled(true);
    }

    @EventHandler(ignoreCancelled = true)
    public void onInteractAt(PlayerInteractAtEntityEvent e) {
        if (flies.isFly(e.getRightClicked())) e.setCancelled(true);
    }

    @EventHandler(ignoreCancelled = true)
    public void onDamage(EntityDamageEvent e) {
        if (!flies.isFly(e.getEntity())) return;
        FlyController fly = flies.get(e.getEntity());
        switch (e.getCause()) {
            // incidenti dovuti al nostro modo di muoverla (teleport ogni tick): non contano
            case SUFFOCATION, FALL, CRAMMING, FLY_INTO_WALL -> {
                e.setCancelled(true);
                return;
            }
            default -> { }
        }
        if (fly == null) return;
        if (e instanceof EntityDamageByEntityEvent hit) {
            // il lato colpito è quello da cui arriva il colpo (il player o la freccia)
            fly.hit(hit.getDamager().getLocation(), hit.getDamage());
        }
        fly.refreshName();
    }

    @EventHandler
    public void onDeath(EntityDeathEvent e) {
        if (!flies.isFly(e.getEntity())) return;
        e.getDrops().clear();
        e.setDroppedExp(0);
        FlyController fly = flies.get(e.getEntity());
        if (fly != null) fly.onDeath();
    }

    @EventHandler(ignoreCancelled = true)
    public void onPickup(EntityPickupItemEvent e) {
        if (flies.isFly(e.getEntity())) e.setCancelled(true);
    }
}
