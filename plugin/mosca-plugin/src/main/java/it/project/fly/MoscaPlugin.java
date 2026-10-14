package it.project.fly;

import org.bukkit.NamespacedKey;
import org.bukkit.command.PluginCommand;
import org.bukkit.plugin.java.JavaPlugin;

/**
 * The "Mosca" plugin: an Allay whose behaviour comes from the fruit-fly connectome,
 * simulated by an external Python process (one brain per fly).
 *
 * The plugin decides nothing on its own: it gathers the senses (where the players are,
 * which odors are around), sends them to the brain and applies the motor commands
 * that come back.
 */
public final class MoscaPlugin extends JavaPlugin {

    public static NamespacedKey FLY_KEY;

    private BrainLink brain;
    private FlyManager flies;
    private NestListener nests;

    @Override
    public void onEnable() {
        saveDefaultConfig();
        FLY_KEY = new NamespacedKey(this, "mosca");

        Settings settings = Settings.load(getConfig(), getLogger());
        Lang lang = new Lang(this, settings.language());
        brain = new BrainLink(this, settings.brainUrl());
        flies = new FlyManager(this, brain, settings, lang, new java.io.File(getDataFolder(), "logs"));
        flies.removeLeftovers();

        getServer().getPluginManager().registerEvents(new FlyListener(flies), this);
        if (settings.nestEnabled()) {
            nests = new NestListener(flies);
            getServer().getPluginManager().registerEvents(nests, this);
        }

        MoscaCommand cmd = new MoscaCommand(flies, brain);
        PluginCommand pc = getCommand("mosca");
        if (pc != null) {
            pc.setExecutor(cmd);
            pc.setTabCompleter(cmd);
        }

        // the heart of the plugin: one step per server tick
        getServer().getScheduler().runTaskTimer(this, flies::tick, 1L, 1L);
        getLogger().info("Mosca ready. Brain expected at " + settings.brainUrl()
                + " - configured odors: " + settings.odors().size());
    }

    @Override
    public void onDisable() {
        if (nests != null) nests.shutdown();
        if (flies != null) flies.removeAll();
        if (brain != null) brain.stop();
    }
}
