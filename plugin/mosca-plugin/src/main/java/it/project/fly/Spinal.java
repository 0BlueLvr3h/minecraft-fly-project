package it.project.fly;

import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Il "midollo" della mosca.
 *
 * Nella mosca vera il ritmo della camminata lo genera il cordone nervoso
 * ventrale (VNC), che non fa parte del connettoma FlyWire. Qui lo sostituiamo:
 *
 *  - camminata spontanea: alterna pause e passeggiate, con una sterzata che
 *    deriva lentamente (come una mosca che esplora)
 *  - odori: quando il CERVELLO percepisce un odore (valore 0..1 dal server),
 *    gira verso la sorgente se l'odore è attraente, via se è avversivo.
 *    Più il cervello lo sente forte, più l'odore prende il sopravvento.
 *
 * La fuga NON passa da qui: la decide interamente il cervello.
 */
public final class Spinal {

    /** Una sorgente di odore vista dalla mosca: angolo in gradi (+ = sinistra), distanza. */
    public record Source(String odor, double angle, double dist) {}

    /** Comando per il corpo: avanti (0..1) e sterzata (-1..1, + = sinistra). */
    public record Drive(double fwd, double turn, String mode) {}

    private static final double ARRIVED = 1.2;       // blocchi: "sono arrivata alla frutta"
    private static final double TURN_GAIN = 45.0;    // gradi di errore per sterzata piena

    private final Settings cfg;
    private final Random rng = new Random();

    private boolean walking;
    private int phaseTicks;
    private double wanderTurn;

    public Spinal(Settings cfg) {
        this.cfg = cfg;
    }

    public Drive step(Map<String, Double> perceive, List<Source> sources, boolean blocked) {
        // --- camminata spontanea -------------------------------------------
        if (--phaseTicks <= 0) {
            walking = rng.nextDouble() < cfg.activity();
            phaseTicks = 40 + rng.nextInt(100);       // 2..7 secondi
        }
        wanderTurn += -0.05 * wanderTurn + rng.nextGaussian() * 0.08;
        wanderTurn = clamp(wanderTurn, -1, 1);
        double wFwd = walking ? cfg.wanderSpeed() : 0.0;
        double wTurn = walking ? wanderTurn : wanderTurn * 0.3;   // da ferma si guarda intorno

        // --- odori: il più forte tra quelli che il cervello percepisce --------
        double best = 0, oFwd = 0, oTurn = 0;
        String bestOdor = null;
        if (perceive != null) {
            for (Map.Entry<String, Double> e : perceive.entrySet()) {
                double v = cfg.valence().getOrDefault(e.getKey(), 0.0);
                double strength = e.getValue() * Math.abs(v);
                if (strength < 0.05 || strength <= best) continue;
                Source src = nearest(sources, e.getKey());
                if (src == null) continue;

                best = strength;
                bestOdor = e.getKey();
                if (v > 0) {                          // attraente: punta verso
                    oTurn = clamp(src.angle() / TURN_GAIN, -1, 1);
                    oFwd = src.dist() > ARRIVED ? 1.0 - 0.5 * Math.abs(oTurn) : 0.0;
                } else {                              // avversivo: punta dalla parte opposta
                    double away = src.angle() > 0 ? src.angle() - 180 : src.angle() + 180;
                    oTurn = clamp(away / TURN_GAIN, -1, 1);
                    oFwd = 1.0 - 0.5 * Math.abs(oTurn);
                }
            }
        }

        // --- miscela: più l'odore è forte, più comanda lui ------------------
        double w = clamp(best * 1.5, 0, 1);
        double fwd = wFwd * (1 - w) + oFwd * w;
        double turn = wTurn * (1 - w) + oTurn * w;
        String mode = bestOdor == null ? (walking ? "passeggia" : "ferma")
                : (cfg.valence().getOrDefault(bestOdor, 0.0) > 0 ? "segue " : "evita ") + bestOdor;

        if (blocked) {                                // davanti c'è un muro: gira
            turn = wanderTurn >= 0 ? 1.0 : -1.0;
            fwd *= 0.3;
        }
        return new Drive(fwd, turn, mode);
    }

    private static Source nearest(List<Source> sources, String odor) {
        Source best = null;
        for (Source s : sources) {
            if (s.odor().equals(odor) && (best == null || s.dist() < best.dist())) best = s;
        }
        return best;
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
