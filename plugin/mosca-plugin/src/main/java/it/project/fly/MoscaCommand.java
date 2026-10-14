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

/** /mosca spawn [force] | remove [all] | tp [number] | status | debug */
public final class MoscaCommand implements CommandExecutor, TabCompleter {

    private final FlyManager flies;
    private final BrainLink brain;
    private final Lang lang;

    public MoscaCommand(FlyManager flies, BrainLink brain) {
        this.flies = flies;
        this.brain = brain;
        this.lang = flies.lang;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        String sub = args.length > 0 ? args[0].toLowerCase(Locale.ROOT) : "status";
        String opt = args.length > 1 ? args[1].toLowerCase(Locale.ROOT) : "";
        switch (sub) {
            case "spawn" -> {
                if (!(sender instanceof Player p)) {
                    msg(sender, lang.get("command.spawn.players_only"), NamedTextColor.RED);
                    return true;
                }
                Vector dir = p.getLocation().getDirection().setY(0);
                if (dir.lengthSquared() < 1e-6) dir = new Vector(0, 0, 1);
                Location loc = p.getEyeLocation().add(dir.normalize().multiply(3));
                loc.setYaw(p.getLocation().getYaw() + 180f);   // facing you
                loc.setPitch(0);
                boolean force = opt.equals("force");
                FlyManager.Refusal no = force ? null : flies.refusal(p.getUniqueId());
                FlyController fly = no == null ? flies.spawn(loc, p.getUniqueId(), force) : null;
                if (fly == null) {
                    msg(sender, lang.get("command.spawn.refused", "reason",
                            flies.describe(no == null ? FlyManager.Refusal.PC_LIMIT : no, p.getUniqueId())),
                            NamedTextColor.RED);
                } else {
                    msg(sender, lang.get("command.spawn.done", "id", fly.id(), "count", flies.count()),
                            NamedTextColor.GREEN);
                }
            }
            case "tp" -> {
                if (!(sender instanceof Player p)) {
                    msg(sender, lang.get("command.tp.players_only"), NamedTextColor.RED);
                    return true;
                }
                // without a number: your fly (the one you spawned, or the one bonded to you)
                FlyController fly = !opt.isEmpty() ? flies.byId(opt) : flies.ownedBy(p.getUniqueId());
                if (fly == null && opt.isEmpty()) fly = flies.bondedTo(p.getUniqueId());
                if (fly == null) {
                    msg(sender, opt.isEmpty() ? lang.get("command.tp.none_own")
                            : lang.get("command.tp.not_found", "id", opt), NamedTextColor.RED);
                } else if (fly.teleportTo(p)) {
                    msg(sender, lang.get("command.tp.done", "id", fly.id()), NamedTextColor.GREEN);
                } else {
                    msg(sender, lang.get("command.tp.no_space"), NamedTextColor.RED);
                }
            }
            case "debug" -> {
                if (!(sender instanceof Player p)) {
                    msg(sender, lang.get("command.debug.players_only"), NamedTextColor.RED);
                    return true;
                }
                boolean on = flies.toggleDebug(p);
                String file = flies.logFile() != null ? flies.logFile().getName() : "?";
                if (on) {
                    msg(sender, lang.get("command.debug.enabled", "file", file), NamedTextColor.GREEN);
                } else {
                    msg(sender, lang.get("command.debug.disabled", "file", file), NamedTextColor.YELLOW);
                }
            }
            case "remove" -> {
                if (opt.equals("all")) {
                    int n = flies.count();
                    flies.removeAll();
                    msg(sender, lang.get("command.remove.all", "count", n), NamedTextColor.YELLOW);
                    return true;
                }
                FlyController fly = sender instanceof Player p ? flies.nearest(p.getLocation()) : null;
                if (fly == null) {
                    msg(sender, lang.get("command.remove.none_near"), NamedTextColor.GRAY);
                    return true;
                }
                flies.remove(fly);
                msg(sender, lang.get("command.remove.done", "id", fly.id()), NamedTextColor.YELLOW);
            }
            case "status" -> {
                FlyManager.Refusal pc = flies.refusal(null);
                String births = pc == null ? lang.get("status.births_yes") : switch (pc) {
                    case WAIT -> lang.get("status.births_wait");
                    case MAX -> lang.get("status.births_max");
                    default -> lang.get("status.births_pc");
                };
                msg(sender, lang.get("status.summary",
                                "count", flies.count(),
                                "brains", lang.get(brain.isConnected() ? "status.brains_connected" : "status.brains_disconnected"),
                                "rt", f("%.2f", brain.rtAvg()),
                                "births", births),
                        brain.isConnected() || flies.count() == 0 ? NamedTextColor.AQUA : NamedTextColor.RED);
                if (sender instanceof Player p && flies.cfg.maxPerPlayer() > 0) {
                    msg(sender, lang.get("status.own", "owned", flies.owned(p.getUniqueId()), "max", flies.cfg.maxPerPlayer()),
                            NamedTextColor.AQUA);
                }
                FlyController fly = sender instanceof Player p ? flies.nearest(p.getLocation()) : null;
                if (fly == null) return true;
                BrainLink.Motor m = brain.latest(fly.id());
                msg(sender, lang.get("status.nearest", "id", fly.id(), "mode", fly.modeLabel(),
                        "health", f("%.0f", fly.health()), "takeoffs", fly.takeoffs()), NamedTextColor.AQUA);
                if (m != null) {
                    msg(sender, lang.get("status.brain",
                            "loomL", f("%.2f", m.loomL()), "loomR", f("%.2f", m.loomR()),
                            "gfL", m.gfL(), "gfR", m.gfR(), "mn9", f("%.0f", m.mn9()),
                            "groom", f("%.0f", m.groom()), "habituation", f("%.0f", 100 * (1 - m.loomGain()))),
                            NamedTextColor.GRAY);
                    msg(sender, lang.get("status.memory", "familiar", f("%.2f", m.memFamiliar()),
                            "learned", f("%+.2f", fly.memory()), "now", f("%+.2f", m.memValence())), NamedTextColor.GRAY);
                    java.util.UUID fav = fly.favorite();
                    org.bukkit.OfflinePlayer fp = fav != null ? org.bukkit.Bukkit.getOfflinePlayer(fav) : null;
                    String you = sender instanceof Player pl
                            ? lang.get("status.favorite_you", "liking", f("%+.2f", fly.likingOf(pl.getUniqueId()))) : "";
                    String who = fp != null && fp.getName() != null
                            ? lang.get("status.favorite_player", "name", fp.getName(), "liking", f("%+.2f", fly.likingOf(fav)))
                            : lang.get("status.favorite_nobody");
                    msg(sender, lang.get("status.favorite", "who", who, "you", you), NamedTextColor.GRAY);
                    StringBuilder p = new StringBuilder();
                    m.perceive().forEach((k, v) -> p.append(String.format(Locale.ROOT, " %s %.2f", k, v)));
                    msg(sender, lang.get("status.odors", "odors", p.toString()), NamedTextColor.GRAY);
                }
            }
            default -> msg(sender, lang.get("command.usage"), NamedTextColor.RED);
        }
        return true;
    }

    @Override
    public List<String> onTabComplete(CommandSender sender, Command command, String alias, String[] args) {
        if (args.length == 1) {
            String a = args[0].toLowerCase(Locale.ROOT);
            return List.of("spawn", "remove", "status", "debug", "tp").stream().filter(s -> s.startsWith(a)).toList();
        }
        if (args.length == 2) {
            String a = args[1].toLowerCase(Locale.ROOT);
            List<String> opts = switch (args[0].toLowerCase(Locale.ROOT)) {
                case "spawn" -> List.of("force");
                case "remove" -> List.of("all");
                case "tp" -> flies.all().stream().map(FlyController::id).toList();
                default -> List.of();
            };
            return opts.stream().filter(s -> s.startsWith(a)).toList();
        }
        return List.of();
    }

    private static void msg(CommandSender s, String text, NamedTextColor color) {
        s.sendMessage(Component.text(text, color));
    }

    /** Formats a number the same way regardless of the server locale. */
    private static String f(String format, double v) {
        return String.format(Locale.ROOT, format, v);
    }
}
