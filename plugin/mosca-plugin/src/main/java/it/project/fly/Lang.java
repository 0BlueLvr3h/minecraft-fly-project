package it.project.fly;

import org.bukkit.configuration.ConfigurationSection;
import org.bukkit.configuration.file.YamlConfiguration;
import org.bukkit.plugin.java.JavaPlugin;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

/**
 * In-game messages in the language chosen in config.yml ({@code language: en | it | ...}).
 *
 * Any language works if plugins/Mosca/lang/&lt;language&gt;.yml exists (copy en.yml and translate it).
 *
 * Both language files ship inside the jar (lang/en.yml, lang/it.yml) and are copied to
 * plugins/Mosca/lang/ on first start, so server owners can edit them. A key is looked up in:
 *   1. plugins/Mosca/lang/&lt;language&gt;.yml (the editable copy)
 *   2. the same language file inside the jar (keys added by a newer plugin version)
 *   3. the English file inside the jar
 *   4. the key itself
 *
 * Messages may contain {name} placeholders, filled by {@link #get(String, Object...)}.
 */
public final class Lang {

    /** Languages shipped with the plugin. */
    public static final List<String> SUPPORTED = List.of("en", "it");
    public static final String DEFAULT = "en";

    private final String language;
    private final Logger log;
    private final YamlConfiguration user;      // plugins/Mosca/lang/<language>.yml (may be null)
    private final YamlConfiguration bundled;   // lang/<language>.yml inside the jar (may be null)
    private final YamlConfiguration english;   // lang/en.yml inside the jar (may be null)
    private final Set<String> warned = new HashSet<>();

    public Lang(JavaPlugin plugin, String language) {
        this.log = plugin.getLogger();

        // copy the bundled files once, without overwriting the owner's edits
        for (String code : SUPPORTED) {
            String path = "lang/" + code + ".yml";
            if (new File(plugin.getDataFolder(), path).exists()) continue;
            try {
                plugin.saveResource(path, false);
            } catch (IllegalArgumentException e) {
                log.warning("Language file missing from the jar: " + path);
            }
        }

        // a shipped language, or any other one the owner added as plugins/Mosca/lang/<code>.yml
        boolean exists = SUPPORTED.contains(language)
                || new File(plugin.getDataFolder(), "lang/" + language + ".yml").exists();
        if (!exists) log.warning("No file plugins/Mosca/lang/" + language + ".yml: using " + DEFAULT);
        this.language = exists ? language : DEFAULT;

        File file = new File(plugin.getDataFolder(), "lang/" + this.language + ".yml");
        this.user = file.exists() ? YamlConfiguration.loadConfiguration(file) : null;
        this.bundled = SUPPORTED.contains(this.language) ? loadBundled(plugin, this.language) : null;
        this.english = DEFAULT.equals(this.language) ? bundled : loadBundled(plugin, DEFAULT);
        log.info("Message language: " + this.language + " (plugins/Mosca/lang/" + this.language + ".yml)");
    }

    public String language() {
        return language;
    }

    /**
     * The message for this key, with {name} placeholders replaced. kv are name/value pairs,
     * e.g. get("fly.died", "name", "Mosca", "id", "3").
     */
    public String get(String key, Object... kv) {
        String text = lookup(user, key);
        if (text == null) text = lookup(bundled, key);
        if (text == null) text = lookup(english, key);
        if (text == null) {
            if (warned.add(key)) log.warning("Missing message key: " + key);
            text = key;
        }
        return kv.length < 2 ? text : fill(text, kv);
    }

    /** Replaces {name} placeholders in a single pass (values are never re-scanned). */
    private static String fill(String text, Object[] kv) {
        Map<String, String> values = new HashMap<>();
        for (int i = 0; i + 1 < kv.length; i += 2) values.put(String.valueOf(kv[i]), String.valueOf(kv[i + 1]));
        StringBuilder out = new StringBuilder(text.length() + 16);
        int i = 0;
        while (i < text.length()) {
            char c = text.charAt(i);
            int close = c == '{' ? text.indexOf('}', i + 1) : -1;
            if (close > i) {
                String v = values.get(text.substring(i + 1, close));
                if (v != null) {
                    out.append(v);
                    i = close + 1;
                    continue;
                }
            }
            out.append(c);
            i++;
        }
        return out.toString();
    }

    private static String lookup(ConfigurationSection c, String key) {
        if (c == null) return null;
        Object v = c.get(key);
        return v == null || v instanceof ConfigurationSection ? null : v.toString();
    }

    private YamlConfiguration loadBundled(JavaPlugin plugin, String code) {
        String path = "lang/" + code + ".yml";
        InputStream in = plugin.getResource(path);
        if (in == null) {
            log.warning("Language file missing from the jar: " + path);
            return null;
        }
        try (Reader r = new InputStreamReader(in, StandardCharsets.UTF_8)) {
            return YamlConfiguration.loadConfiguration(r);
        } catch (IOException e) {
            log.warning("Cannot read " + path + " from the jar: " + e.getMessage());
            return null;
        }
    }
}
