package it.project.fly;

import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.format.NamedTextColor;
import org.bukkit.Location;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.command.TabCompleter;
import org.bukkit.entity.Player;
import org.bukkit.util.Vector;

import java.util.List;
import java.util.Locale;

/** /mosca spawn [force] | remove [all] | status | debug */
public final class MoscaCommand implements CommandExecutor, TabCompleter {

    private final FlyManager flies;
    private final BrainLink brain;

    public MoscaCommand(FlyManager flies, BrainLink brain) {
        this.flies = flies;
        this.brain = brain;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        String sub = args.length > 0 ? args[0].toLowerCase(Locale.ROOT) : "status";
        String opt = args.length > 1 ? args[1].toLowerCase(Locale.ROOT) : "";
        switch (sub) {
            case "spawn" -> {
                if (!(sender instanceof Player p)) {
                    msg(sender, "Solo un player può far nascere una mosca.", NamedTextColor.RED);
                    return true;
                }
                Vector dir = p.getLocation().getDirection().setY(0);
                if (dir.lengthSquared() < 1e-6) dir = new Vector(0, 0, 1);
                Location loc = p.getEyeLocation().add(dir.normalize().multiply(3));
                loc.setYaw(p.getLocation().getYaw() + 180f);   // guarda verso di te
                loc.setPitch(0);
                boolean force = opt.equals("force");
                FlyManager.Refusal no = force ? null : flies.refusal(p.getUniqueId());
                FlyController fly = no == null ? flies.spawn(loc, p.getUniqueId(), force) : null;
                if (fly == null) {
                    msg(sender, "Nessuna nuova mosca: "
                            + flies.describe(no == null ? FlyManager.Refusal.PC_LIMIT : no, p.getUniqueId())
                            + " Usa /mosca spawn force per farla nascere comunque.", NamedTextColor.RED);
                } else {
                    msg(sender, "È nata la mosca " + fly.id() + " (" + flies.count() + " in tutto).",
                            NamedTextColor.GREEN);
                }
            }
            case "debug" -> {
                if (!(sender instanceof Player p)) {
                    msg(sender, "Il debug si vede nell'action bar: solo per i player.", NamedTextColor.RED);
                    return true;
                }
                boolean on = flies.toggleDebug(p);
                String file = flies.logFile() != null ? flies.logFile().getName() : "?";
                if (on) {
                    msg(sender, "Debug attivo (mostra la mosca più vicina), registro: plugins/Mosca/logs/" + file,
                            NamedTextColor.GREEN);
                } else {
                    msg(sender, "Debug spento. Registro salvato: plugins/Mosca/logs/" + file, NamedTextColor.YELLOW);
                }
            }
            case "remove" -> {
                if (opt.equals("all")) {
                    int n = flies.count();
                    flies.removeAll();
                    msg(sender, "Rimosse " + n + " mosche.", NamedTextColor.YELLOW);
                    return true;
                }
                FlyController fly = sender instanceof Player p ? flies.nearest(p.getLocation()) : null;
                if (fly == null) {
                    msg(sender, "Nessuna mosca vicina. Per toglierle tutte: /mosca remove all", NamedTextColor.GRAY);
                    return true;
                }
                flies.remove(fly);
                msg(sender, "Rimossa la mosca " + fly.id() + ".", NamedTextColor.YELLOW);
            }
            case "status" -> {
                FlyManager.Refusal pc = flies.refusal(null);
                String births = pc == null ? "sì" : switch (pc) {
                    case WAIT -> "tra qualche secondo";
                    case MAX -> "no, massimo del server";
                    default -> "no, PC al limite";
                };
                msg(sender, String.format(Locale.ROOT, "Mosche: %d   cervelli: %s   tempo reale medio: %.2f   nuove nascite: %s",
                                flies.count(), brain.isConnected() ? "collegati" : "NON collegati", brain.rtAvg(), births),
                        brain.isConnected() || flies.count() == 0 ? NamedTextColor.AQUA : NamedTextColor.RED);
                if (sender instanceof Player p && flies.cfg.maxPerPlayer() > 0) {
                    msg(sender, "Le tue mosche: " + flies.owned(p.getUniqueId()) + " / " + flies.cfg.maxPerPlayer(),
                            NamedTextColor.AQUA);
                }
                FlyController fly = sender instanceof Player p ? flies.nearest(p.getLocation()) : null;
                if (fly == null) return true;
                BrainLink.Motor m = brain.latest(fly.id());
                msg(sender, String.format(Locale.ROOT, "Più vicina: mosca %s, %s, vita %.0f, decolli %d",
                        fly.id(), fly.mode(), fly.health(), fly.takeoffs()), NamedTextColor.AQUA);
                if (m != null) {
                    msg(sender, String.format(Locale.ROOT,
                            "Looming L/R: %.2f / %.2f   Giant Fiber L/R: %d / %d   MN9 %.0f Hz   pulizia %.0f Hz",
                            m.loomL(), m.loomR(), m.gfL(), m.gfR(), m.mn9(), m.groom()), NamedTextColor.GRAY);
                    msg(sender, String.format(Locale.ROOT,
                            "Memoria: odore di player sentito %.2f, valenza imparata %+.2f (adesso %+.2f)",
                            m.memFamiliar(), fly.memory(), m.memValence()), NamedTextColor.GRAY);
                    StringBuilder p = new StringBuilder("Percezione odori:");
                    m.perceive().forEach((k, v) -> p.append(String.format(Locale.ROOT, " %s %.2f", k, v)));
                    msg(sender, p.toString(), NamedTextColor.GRAY);
                }
            }
            default -> msg(sender, "Uso: /mosca <spawn [force]|remove [all]|status|debug>", NamedTextColor.RED);
        }
        return true;
    }

    @Override
    public List<String> onTabComplete(CommandSender sender, Command command, String alias, String[] args) {
        if (args.length == 1) {
            String a = args[0].toLowerCase(Locale.ROOT);
            return List.of("spawn", "remove", "status", "debug").stream().filter(s -> s.startsWith(a)).toList();
        }
        if (args.length == 2) {
            String a = args[1].toLowerCase(Locale.ROOT);
            List<String> opts = switch (args[0].toLowerCase(Locale.ROOT)) {
                case "spawn" -> List.of("force");
                case "remove" -> List.of("all");
                default -> List.of();
            };
            return opts.stream().filter(s -> s.startsWith(a)).toList();
        }
        return List.of();
    }

    private static void msg(CommandSender s, String text, NamedTextColor color) {
        s.sendMessage(Component.text(text, color));
    }
}
