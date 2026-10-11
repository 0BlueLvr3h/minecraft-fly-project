package it.luca.mosca;

import org.bukkit.NamespacedKey;
import org.bukkit.command.PluginCommand;
import org.bukkit.plugin.java.JavaPlugin;

/**
 * Plugin "Mosca": Allay il cui comportamento arriva dal connettoma della mosca
 * della frutta, simulato da un processo Python esterno (un cervello per mosca).
 *
 * Il plugin non decide niente: raccoglie i sensi (dove sono i player,
 * che odori ci sono intorno), li manda al cervello e applica i comandi
 * motori che tornano indietro.
 */
public final class MoscaPlugin extends JavaPlugin {

    public static NamespacedKey FLY_KEY;

    private BrainLink brain;
    private FlyManager flies;

    @Override
    public void onEnable() {
        saveDefaultConfig();
        FLY_KEY = new NamespacedKey(this, "mosca");

        Settings settings = Settings.load(getConfig(), getLogger());
        brain = new BrainLink(this, settings.brainUrl());
        flies = new FlyManager(this, brain, settings, new java.io.File(getDataFolder(), "logs"));
        flies.removeLeftovers();

        getServer().getPluginManager().registerEvents(new FlyListener(flies), this);

        MoscaCommand cmd = new MoscaCommand(flies, brain);
        PluginCommand pc = getCommand("mosca");
        if (pc != null) {
            pc.setExecutor(cmd);
            pc.setTabCompleter(cmd);
        }

        // il cuore del plugin: un passo per ogni tick del server
        getServer().getScheduler().runTaskTimer(this, flies::tick, 1L, 1L);
        getLogger().info("Mosca pronta. Cervello atteso su " + settings.brainUrl()
                + " - odori configurati: " + settings.odors().size());
    }

    @Override
    public void onDisable() {
        if (flies != null) flies.removeAll();
        if (brain != null) brain.stop();
    }
}
