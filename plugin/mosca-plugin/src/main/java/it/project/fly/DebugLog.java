package it.project.fly;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Debug CSV log: one row per tick (and per nearby player).
 * Written to plugins/Mosca/logs/debug-YYYYMMDD-HHMMSS.csv
 * The "mode" column holds the internal English mode key (see Spinal.Drive), not the translated text.
 */
final class DebugLog implements AutoCloseable {

    static final String HEADER = String.join(",",
            "fly", "t_ms", "tick", "mode", "x", "y", "z", "yaw",
            "brain", "rt", "spikes", "loomL", "loomR", "gfL", "gfR", "takeoff",
            "esc_fwd", "esc_left", "walk_fwd", "walk_turn", "perc_fruity", "perc_rotten",
            "player", "dist", "speed", "angle", "held_odor",
            "taste", "mn9", "feeding", "touchL", "touchR", "groom", "grooming", "health",
            "pain", "mem_valence", "mem_smooth", "mem_familiar", "loom_gain", "bonded");

    private final BufferedWriter out;
    private final File file;
    private final long start = System.currentTimeMillis();
    private int rows;

    DebugLog(File dir) throws IOException {
        if (!dir.exists() && !dir.mkdirs()) throw new IOException("cannot create " + dir);
        String stamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
        file = new File(dir, "debug-" + stamp + ".csv");
        out = Files.newBufferedWriter(file.toPath(), StandardCharsets.UTF_8);
        out.write(HEADER);
        out.newLine();
    }

    File file() {
        return file;
    }

    long elapsed() {
        return System.currentTimeMillis() - start;
    }

    void row(String line) {
        try {
            out.write(line);
            out.newLine();
            if (++rows % 40 == 0) out.flush();          // about every 2 seconds
        } catch (IOException ignored) {
            // debugging must never block the game
        }
    }

    @Override
    public void close() {
        try {
            out.close();
        } catch (IOException ignored) {
        }
    }
}
