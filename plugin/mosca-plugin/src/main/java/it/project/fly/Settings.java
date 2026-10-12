package it.project.fly;

import org.bukkit.Material;
import org.bukkit.configuration.ConfigurationSection;
import org.bukkit.configuration.file.FileConfiguration;

import java.util.EnumMap;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.List;
import java.util.Set;
import java.util.EnumSet;
import java.util.logging.Logger;

/** Impostazioni lette da config.yml (con valori di default se mancano). */
public record Settings(
        String brainUrl,
        double range,
        double maxSpeed,
        double maxTurn,
        double smoothing,
        double hoverHeight,
        double wanderSpeed,
        double activity,
        double hitReflex,
        double healPerBite,
        int maxPerPlayer,
        boolean nestEnabled,
        int hatchSeconds,
        Set<Material> nestFruits,
        Map<String, Double> valence,
        Map<Material, String> odors) {

    public static Settings load(FileConfiguration c, Logger log) {
        Map<Material, String> odors = new EnumMap<>(Material.class);
        ConfigurationSection sec = c.getConfigurationSection("odors");
        if (sec == null) {
            // config.yml vecchio, senza la sezione odori: usiamo questi
            odors.put(Material.APPLE, "fruity");
            odors.put(Material.SWEET_BERRIES, "fruity");
            odors.put(Material.MELON_SLICE, "fruity");
            odors.put(Material.ROTTEN_FLESH, "rotten");
            odors.put(Material.SPIDER_EYE, "rotten");
            log.info("Nessuna sezione 'odors' in config.yml: uso gli odori di default");
        } else {
            for (String key : sec.getKeys(false)) {
                Material m = Material.matchMaterial(key.toUpperCase(Locale.ROOT));
                String odor = sec.getString(key);
                if (m == null || odor == null) {
                    log.warning("Odore ignorato in config.yml: " + key);
                    continue;
                }
                odors.put(m, odor);
            }
        }

        Map<String, Double> valence = new HashMap<>();
        ConfigurationSection val = c.getConfigurationSection("valence");
        if (val == null) {
            valence.put("fruity", 1.0);
            valence.put("rotten", -1.0);
        } else {
            for (String key : val.getKeys(false)) {
                valence.put(key, clamp(val.getDouble(key, 0.0), -1, 1));
            }
        }

        // Nido: la frutta che, da sola, riempie una compostiera e fa nascere una mosca
        Set<Material> fruits = EnumSet.noneOf(Material.class);
        List<String> fruitNames = c.isList("nest.fruits") ? c.getStringList("nest.fruits")
                : List.of("APPLE", "MELON_SLICE", "MELON", "PUMPKIN", "SWEET_BERRIES", "GLOW_BERRIES");
        for (String name : fruitNames) {
            Material m = Material.matchMaterial(name.toUpperCase(Locale.ROOT));
            if (m == null) log.warning("Frutto ignorato in config.yml (nest.fruits): " + name);
            else fruits.add(m);
        }

        return new Settings(
                c.getString("brain-url", "ws://127.0.0.1:8765"),
                c.getDouble("range", 16.0),
                c.getDouble("walk.max-speed", 1.5),
                c.getDouble("walk.max-turn", 120.0),
                clamp(c.getDouble("walk.smoothing", 0.25), 0.01, 1.0),
                c.getDouble("walk.hover-height", 0.6),
                clamp(c.getDouble("spinal.wander-speed", 0.35), 0, 1),
                clamp(c.getDouble("spinal.activity", 0.6), 0, 1),
                clamp(c.getDouble("spinal.hit-reflex", 0.6), 0, 1),
                Math.max(0, c.getDouble("health.heal-per-bite", 2.0)),
                Math.max(0, c.getInt("spawn.max-per-player", 1)),
                c.getBoolean("nest.enabled", true),
                Math.max(1, c.getInt("nest.hatch-seconds", 20)),
                fruits,
                valence,
                odors);
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
