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
 * Turns off the body's vanilla interactions (giving items to the Allay, duplicating it, letting it
 * pick things up). A hit takes away health and becomes touch for the brain; if it dies, it really dies.
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
            // accidents caused by the way we move it (teleport every tick): they don't count
            case SUFFOCATION, FALL, CRAMMING, FLY_INTO_WALL -> {
                e.setCancelled(true);
                return;
            }
            default -> { }
        }
        if (fly == null) return;
        if (e instanceof EntityDamageByEntityEvent hit) {
            // the side that is hit is the one the blow comes from (the player or the arrow)
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
