package it.luca.mosca;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Registro CSV del debug: una riga per tick (e per ogni player vicino).
 * Finisce in plugins/Mosca/logs/debug-AAAAMMGG-HHMMSS.csv
 */
final class DebugLog implements AutoCloseable {

    static final String HEADER = String.join(",",
            "fly", "t_ms", "tick", "mode", "x", "y", "z", "yaw",
            "brain", "rt", "spikes", "loomL", "loomR", "gfL", "gfR", "takeoff",
            "esc_fwd", "esc_left", "walk_fwd", "walk_turn", "perc_fruity", "perc_rotten",
            "player", "dist", "speed", "angle", "held_odor",
            "taste", "mn9", "feeding", "touchL", "touchR", "groom", "grooming", "health");

    private final BufferedWriter out;
    private final File file;
    private final long start = System.currentTimeMillis();
    private int rows;

    DebugLog(File dir) throws IOException {
        if (!dir.exists() && !dir.mkdirs()) throw new IOException("impossibile creare " + dir);
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
            if (++rows % 40 == 0) out.flush();          // circa ogni 2 secondi
        } catch (IOException ignored) {
            // il debug non deve mai bloccare il gioco
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
