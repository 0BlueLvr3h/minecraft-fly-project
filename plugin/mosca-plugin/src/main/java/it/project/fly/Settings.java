package it.project.fly;

import org.bukkit.Material;
import org.bukkit.entity.Allay;
import org.bukkit.entity.Entity;
import org.bukkit.entity.EntityType;
import org.bukkit.entity.Mob;
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

/** Settings read from config.yml (with default values when missing). */
public record Settings(
        String language,
        String brainUrl,
        Class<? extends Mob> bodyClass,
        String bodyName,
        boolean bodyBaby,
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
        double memoryMin,
        double followValence,
        double followDistance,
        double keepDistance,
        double teleportDistance,
        double teleportValence,
        double flightChance,
        double flightSpeed,
        double flightHeight,
        double flyToOdorDistance,
        Map<String, Double> valence,
        Map<Material, String> odors) {

    public static Settings load(FileConfiguration c, Logger log) {
        // Language of in-game messages (lang/<language>.yml)
        String rawLanguage = c.getString("language", Lang.DEFAULT);
        String language = rawLanguage == null ? Lang.DEFAULT : rawLanguage.trim().toLowerCase(Locale.ROOT);
        if (!language.matches("[a-z][a-z_-]{1,15}")) {           // file name lang/<language>.yml
            log.warning("language: " + language + " is not a valid language code, using " + Lang.DEFAULT);
            language = Lang.DEFAULT;
        }

        Map<Material, String> odors = new EnumMap<>(Material.class);
        ConfigurationSection sec = c.getConfigurationSection("odors");
        if (sec == null) {
            // old config.yml without the odors section: use these
            odors.put(Material.APPLE, "fruity");
            odors.put(Material.SWEET_BERRIES, "fruity");
            odors.put(Material.MELON_SLICE, "fruity");
            odors.put(Material.ROTTEN_FLESH, "rotten");
            odors.put(Material.SPIDER_EYE, "rotten");
            log.info("No 'odors' section in config.yml: using the default odors");
        } else {
            for (String key : sec.getKeys(false)) {
                Material m = Material.matchMaterial(key.toUpperCase(Locale.ROOT));
                String odor = sec.getString(key);
                if (m == null || odor == null) {
                    log.warning("Ignoring odor in config.yml: " + key);
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

        // Nest: the fruit that, on its own, fills a composter and hatches a fly
        Set<Material> fruits = EnumSet.noneOf(Material.class);
        List<String> fruitNames = c.isList("nest.fruits") ? c.getStringList("nest.fruits")
                : List.of("APPLE", "MELON_SLICE", "MELON", "PUMPKIN", "SWEET_BERRIES", "GLOW_BERRIES");
        for (String name : fruitNames) {
            Material m = Material.matchMaterial(name.toUpperCase(Locale.ROOT));
            if (m == null) log.warning("Ignoring fruit in config.yml (nest.fruits): " + name);
            else fruits.add(m);
        }

        // Body: any mob (ALLAY, BEE, PARROT...). The brain is still the Drosophila one.
        Class<? extends Mob> body = Allay.class;
        String bodyType = c.getString("body.entity", "ALLAY");
        try {
            Class<? extends Entity> k = EntityType.valueOf(bodyType.toUpperCase(Locale.ROOT)).getEntityClass();
            if (k != null && Mob.class.isAssignableFrom(k)) body = k.asSubclass(Mob.class);
            else log.warning("body.entity: " + bodyType + " is not a mob, using ALLAY");
        } catch (IllegalArgumentException e) {
            log.warning("body.entity: " + bodyType + " is unknown, using ALLAY");
        }

        return new Settings(
                language,
                c.getString("brain-url", "ws://127.0.0.1:8765"),
                body,
                c.getString("body.name", "Mosca"),
                c.getBoolean("body.baby", false),
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
                clamp(c.getDouble("memory.min-valence", 0.1), 0, 1),
                clamp(c.getDouble("memory.follow-valence", 0.15), 0, 2),
                Math.max(1.0, c.getDouble("memory.follow-distance", 4.0)),
                Math.max(0.5, c.getDouble("memory.keep-distance", 2.0)),
                Math.max(0, c.getDouble("memory.teleport-distance", 20.0)),
                clamp(c.getDouble("memory.teleport-valence", 0.2), 0, 2),
                clamp(c.getDouble("flight.chance", 0.004), 0, 1),
                Math.max(0.5, c.getDouble("flight.speed", 4.0)),
                Math.max(0.5, c.getDouble("flight.height", 2.0)),
                Math.max(2.0, c.getDouble("flight.to-odor-distance", 6.0)),
                valence,
                odors);
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
