# Mosca

Un mob di Minecraft (di default un Allay, ma può essere un'ape o altro) guidato dal
connettoma della mosca della frutta (*Drosophila melanogaster*, FlyWire v783): ogni mosca
in gioco ha un cervello di 139.255 neuroni simulato in tempo reale sul PC.

- **Fuga**: LC4/LPLC2 → Giant Fiber. Scappa se le corri contro, ti lascia avvicinare se sei
  accovacciato. Si **abitua**: se ti muovi spesso vicino a lei senza farle male scappa sempre
  meno; un colpo la rende di nuovo sensibile.
- **Olfatto**: frutta e marcio entrano dalle antenne e arrivano al corno laterale.
- **Gusto**: zucchero → MN9 (proboscide). Mangia la frutta e recupera vita.
- **Tatto**: un colpo stimola le setole della testa; con un colpo forte scatta il riflesso di fuga.
- **Memoria**: riconosce i player dall'odore (ognuno ha il suo) e impara nel corpo fungiforme.
  Se mangia mentre sei vicino ti segue; se la colpisci ti evita.
- **Volo**: esplora in volo con tratti dritti e virate rapide (saccadi), vola verso la frutta
  lontana e dietro ai player che conosce.
- **Camminata**: la genera il plugin (il "midollo"), perché FlyWire non contiene il cordone nervoso.
- **Teletrasporto** (opzionale): se il player che le piace di più si allontana troppo, gli compare accanto.
- **Nido**: una compostiera piena solo di frutta fa nascere una mosca.

Il modello è quello di Shiu et al. (Nature, 2024) con tre correzioni di errori del
connettoma e un adattamento leggero: vedi `RECOMMENDED` in `scripts/flysim.py`.
La memoria aggiunge l'unica plasticità misurata nella mosca vera (sinapsi KC → MBON che
si indeboliscono con la dopamina): vedi `scripts/memory.py`.

## Come funziona

```
  PC (Windows, GPU non necessaria)                 Server Minecraft (Paper 26.3)
  ┌──────────────────────────────┐   WebSocket    ┌──────────────────────────────┐
  │ scripts/02_server.py         │ ◄────────────► │ plugin Mosca                 │
  │  un cervello per mosca       │  sensi/comandi │  il corpo: sensi, camminata, │
  │  esploratore 3D su :8766     │   ogni tick    │  volo, comandi, config.yml   │
  └──────────────────────────────┘                └──────────────────────────────┘
```

Il cervello gira sul PC, il plugin sul server Minecraft. Se sono sulla stessa macchina si
parlano direttamente; se il server è su una VM serve un tunnel SSH (vedi sotto).

## Struttura

```
scripts/              cervello: simulatore, circuiti, memoria, server e script di diagnosi
explorer/             esploratore 3D del cervello (servito dal server su :8766)
plugin/mosca-plugin/  plugin Paper (Java 25, Paper 26.3)
avvia_cervello.bat    avvia il server dei cervelli con un doppio clic
tunnel.example.bat    modello del tunnel verso la VM (copialo in tunnel.bat e completalo)
data/                 dati FlyWire: NON nel repository, vedi sotto
```

## 1. Preparare il cervello (una volta sola)

1. **Dati FlyWire** in `data/raw/`:
   - `proofread_connections_783.feather` e `proofread_root_ids_783.npy` da
     [Zenodo, FlyWire 783](https://zenodo.org/records/10676866)
   - `Supplemental_file1_neuron_annotations.tsv` da
     [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations/tree/main/supplemental_files)
2. **Python** (3.10–3.12), dalla cartella del progetto:
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python scripts\00_prepare.py           (prepara il connettoma, un minuto)
   python scripts\17_odori_player.py      (sceglie gli odori dei player, 5-10 minuti)
   ```
   Se PowerShell non lascia attivare il `.venv`:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

   Il `.venv` va attivato sempre **dalla cartella del progetto**: un `.venv` di un'altra
   cartella non ha le librerie (errore `No module named 'numpy'`).

## 2. Compilare il plugin (il file .jar)

Serve **IntelliJ IDEA** (va bene la Community) e un **JDK 25**.

1. In IntelliJ: *File → Open* e scegli la cartella **`plugin/mosca-plugin`** (quella con
   `build.gradle.kts`, non la cartella `plugin`). Se chiede "Trust project", conferma.
2. Se non hai il JDK 25: *File → Project Structure → Project → SDK → Add SDK → Download JDK*,
   versione 25 (va bene qualsiasi distribuzione, per esempio Temurin).
   Controlla anche *Settings → Build, Execution, Deployment → Build Tools → Gradle →
   Gradle JVM*: deve essere il JDK 25.
3. Aspetta che Gradle finisca di scaricare le dipendenze (barra in basso).
4. Apri il pannello **Gradle** (l'elefante a destra) → *mosca → Tasks → build* →
   doppio clic su **build**.
5. Il plugin è in **`plugin/mosca-plugin/build/libs/mosca-0.1.0.jar`**.

Da riga di comando, con Gradle installato: dalla cartella `plugin/mosca-plugin`,
`gradle build` (o `.\gradlew.bat build` se IntelliJ ha creato il wrapper).

**Problemi frequenti**
- *Could not find io.papermc.paper:paper-api:26.3.+*: in `plugin/mosca-plugin/gradle.properties`
  scrivi la versione esatta delle API (la trovi su
  [repo.papermc.io](https://repo.papermc.io/repository/maven-public/io/papermc/paper/paper-api/),
  per esempio `26.3.build.12-stable`).
- *invalid source release: 25* o simili: Gradle non sta usando il JDK 25 (punto 2).
- Dopo aver rinominato il package: controlla che `main:` in
  `src/main/resources/plugin.yml` sia il nome completo della classe (`it.project.fly.MoscaPlugin`).

## 3. Installare il plugin sul server

1. Copia `mosca-0.1.0.jar` nella cartella **`plugins`** del server Paper
   (con Crafty: *Files → plugins → Upload*). Togli il jar vecchio se c'è.
2. **Riavvia** il server (non `/reload`).
3. Al primo avvio il plugin crea **`plugins/Mosca/config.yml`**. Modificalo (vedi
   [Configurazione](#5-configurazione)) e riavvia di nuovo.

Quando aggiorni il plugin, il `config.yml` già sul server **non viene sovrascritto**: le
voci nuove che mancano usano i valori di default. Per usarle, copiale dal
`config.yml` del progetto (`plugin/mosca-plugin/src/main/resources/config.yml`).

## 4. Collegare cervello e server

### Server Minecraft sullo stesso PC
Non serve niente: `brain-url: "ws://127.0.0.1:8765"` (il default).

### Server Minecraft su una VM (tunnel SSH)
Il PC "presta" la porta 8765 alla VM con un tunnel SSH inverso: non si aprono porte verso casa.
Se il server Minecraft gira in **Docker** (per esempio con Crafty), dal container la VM ha un
indirizzo suo (il *gateway* della rete Docker) e il tunnel deve arrivare lì.

Sulla VM, una volta sola:
```bash
# indirizzo della VM visto dal container (es. 172.18.0.1)
docker inspect -f '{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}' NOME_CONTAINER

# permette al tunnel di ascoltare su quell'indirizzo
echo "GatewayPorts clientspecified" | sudo tee /etc/ssh/sshd_config.d/mosca.conf
sudo systemctl restart ssh

# lascia passare la porta 8765 solo dalle reti Docker (le immagini Ubuntu di Oracle bloccano tutto)
sudo iptables -I INPUT -s 172.16.0.0/12 -p tcp --dport 8765 -j ACCEPT
sudo netfilter-persistent save
```

Sul PC: copia `tunnel.example.bat` in **`tunnel.bat`** e scrivici gateway, chiave e IP
della VM (`tunnel.bat` è escluso da git: contiene i tuoi dati). Nel `config.yml` del server:
`brain-url: "ws://GATEWAY:8765"` (es. `ws://172.18.0.1:8765`).

Controllo dalla VM, con cervello e tunnel accesi: `curl -i http://GATEWAY:8765` deve
rispondere con un codice `400` o `426` (è il server dei cervelli che chiede WebSocket).

## Uso quotidiano

1. `avvia_cervello.bat` (doppio clic): carica il connettoma e aspetta le mosche.
2. `tunnel.bat` se il server è su una VM.
3. Il server Minecraft (l'ordine non è obbligatorio: il plugin si ricollega da solo ogni 5 s).

Esploratore 3D del cervello: <http://127.0.0.1:8766> (scegli la mosca da "Cervello di").

### Comandi (permesso `mosca.admin`, di default gli op)

| Comando | Cosa fa |
|---|---|
| `/mosca spawn [force]` | fa nascere una mosca davanti a te (`force` ignora i limiti) |
| `/mosca tp [numero]` | ti porta vicino la tua mosca, o quella col numero dato |
| `/mosca status` | stato del cervello e della mosca più vicina (memoria, abituazione, a chi è legata) |
| `/mosca debug` | action bar di debug e registro CSV in `plugins/Mosca/logs/` |
| `/mosca remove [all]` | toglie la mosca più vicina, o tutte |

Senza comandi: riempi una compostiera **solo di frutta** e invece della farina d'ossa, dopo
un po', ne esce una mosca.

### Come farsi amica una mosca
Lascia cadere della frutta vicino a lei e resta a 1-2 blocchi mentre mangia: lo zucchero è la
ricompensa e il tuo odore diventa "buono". Poi ti segue, anche in volo. Se la colpisci il tuo
odore diventa "cattivo" e ti evita. La memoria svanisce in circa mezz'ora e muore con la mosca;
se riavvii `02_server.py` i cervelli ripartono da zero.

## 5. Configurazione

Il file è `plugins/Mosca/config.yml` sul server. Dopo una modifica **riavvia** il server.

### Collegamento e corpo
| Voce | Default | Significato |
|---|---|---|
| `brain-url` | `ws://127.0.0.1:8765` | dove sta il server dei cervelli (vedi sezione 4) |
| `body.entity` | `ALLAY` | il mob che fa da corpo: `ALLAY`, `BEE`, `PARROT`, `BAT`... qualsiasi mob |
| `body.name` | `Mosca` | nome sopra la testa (seguito da numero e cuori) |
| `body.baby` | `false` | `true` = cucciolo, per i mob che ce l'hanno (ape, pappagallo...) |
| `range` | `16` | blocchi entro cui vede i player e sente gli odori |

Cambiando corpo, le mosche già in gioco tengono quello vecchio: `/mosca remove all` e falle rinascere.

### Movimento
| Voce | Default | Significato |
|---|---|---|
| `walk.max-speed` | `1.5` | blocchi/s a velocità piena di camminata |
| `walk.max-turn` | `120` | gradi/s a sterzata piena |
| `walk.smoothing` | `0.25` | 0..1, più basso = movimenti più morbidi ma più lenti |
| `walk.hover-height` | `0.6` | quanto resta sollevata da terra (abbassalo per i cuccioli) |
| `spinal.wander-speed` | `0.35` | frazione della velocità quando gironzola |
| `spinal.activity` | `0.6` | 0..1, quanto spesso cammina invece di stare ferma |
| `spinal.hit-reflex` | `0.6` | colpo oltre cui salta via subito (pugno ≈ 0.3, spada ≈ 1; 0 = spento) |
| `flight.chance` | `0.004` | probabilità per tick di decollare da sola (≈ ogni 12 s di gironzolo) |
| `flight.speed` | `4.0` | blocchi/s in volo |
| `flight.height` | `2.0` | quota di volo sopra il terreno |
| `flight.to-odor-distance` | `6.0` | oltre questa distanza vola verso un odore attraente invece di camminare |

### Memoria, seguirti e teletrasporto
| Voce | Default | Significato |
|---|---|---|
| `memory.min-valence` | `0.1` | sotto questa valenza un player le è indifferente |
| `memory.follow-valence` | `0.15` | sopra questa ti segue (e decolla per raggiungerti) |
| `memory.follow-distance` | `4.0` | blocchi oltre i quali decolla per seguirti |
| `memory.keep-distance` | `2.0` | blocchi a cui si ferma vicino a te |
| `memory.teleport-distance` | `20` | oltre questa distanza (o in un altro mondo) compare accanto al player preferito; `0` = spento |
| `memory.teleport-valence` | `0.2` | quanto deve piacerle un player per teletrasportarsi da lui |

### Nascite, nido, vita
| Voce | Default | Significato |
|---|---|---|
| `spawn.max-per-player` | `1` | mosche per player (`0` = nessun limite) |
| `nest.enabled` | `true` | compostiera-nido attiva |
| `nest.hatch-seconds` | `20` | quanto "fermenta" la compostiera piena prima della nascita |
| `nest.fruits` | mela, anguria, zucca, bacche | la frutta che conta per il nido |
| `health.heal-per-bite` | `2.0` | vita recuperata ogni 3 s di pasto (2 = un cuore) |

### Odori
| Voce | Significato |
|---|---|
| `valence` | preferenza innata per ogni odore: `+1` ci va incontro, `-1` lo evita |
| `odors` | quale odore ha ogni item (in mano o a terra): `fruity` o `rotten` |

Gli odori dei player non si configurano qui: li sceglie `scripts/17_odori_player.py`.

### Opzioni del server dei cervelli
`python scripts\02_server.py --help` per l'elenco. Le principali:
`--spawn-rt 1.5` (niente nuove mosche se il tempo reale medio scende sotto), `--max-flies 32`,
`--noise 0` (attività spontanea sui neuroni sensoriali), `--shiu` (modello originale senza correzioni).

## Script di diagnosi

| Script | A cosa serve |
|---|---|
| `00_prepare.py` | prepara connettoma e anagrafica dai dati FlyWire |
| `02_server.py` | il server dei cervelli (e dell'esploratore) |
| `03_finto_player.py` | prova il server senza Minecraft |
| `01`, `04`–`14` | prove, tarature e diagnosi di fuga, odori, gusto e tatto |
| `15_corpo_fungiforme.py` | il corpo fungiforme: codifica degli odori e dopamina |
| `16_condizionamento.py` | esperimento di condizionamento (ricompensa, punizione, estraneo) |
| `17_odori_player.py` | sceglie gli odori dei player (`--riscegli` per rifare solo la scelta) |

## Crediti e licenze

Dati del connettoma: FlyWire Consortium (Dorkenwald et al., Nature 2024; Schlegel et al.,
Nature 2024), licenza **CC BY-NC 4.0**: uso non commerciale con attribuzione.
Modello di base: Shiu et al., "A Drosophila computational brain model reveals sensorimotor
processing", Nature 2024.
