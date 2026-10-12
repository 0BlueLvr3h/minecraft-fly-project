# Mosca

Un Allay di Minecraft guidato dal connettoma della mosca della frutta
(*Drosophila melanogaster*, FlyWire v783): ogni mosca in gioco ha un cervello di
139.255 neuroni simulato in tempo reale sul PC.

- **Fuga**: LC4/LPLC2 → Giant Fiber. Scappa se le corri contro, ti lascia avvicinare se sei accovacciato.
- **Olfatto**: frutta e marcio entrano dalle antenne e arrivano al corno laterale.
- **Gusto**: zucchero → MN9 (proboscide). Mangia la frutta e recupera vita.
- **Tatto**: un colpo stimola le setole della testa; con un colpo forte scatta il riflesso di fuga.
- **Nido**: una compostiera piena solo di frutta fa nascere una mosca.
- **Camminata**: la genera il plugin (il "midollo"), perché FlyWire non contiene il cordone nervoso.

Il modello è quello di Shiu et al. (Nature, 2024) con tre correzioni di errori del
connettoma e un adattamento leggero: vedi `RECOMMENDED` in `scripts/flysim.py`.

## Struttura

```
scripts/        cervello: simulatore, circuiti, server e script di diagnosi
explorer/       esploratore 3D del cervello (servito dal server su :8766)
plugin/         plugin Paper (Java 25, Paper 26.3)
data/           dati FlyWire: NON nel repository, vedi sotto
```

## Installazione

1. **Dati FlyWire** in `data/raw/`:
   - `proofread_connections_783.feather` e `proofread_root_ids_783.npy` da
     [Zenodo, FlyWire 783](https://zenodo.org/records/10676866)
   - `Supplemental_file1_neuron_annotations.tsv` da
     [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations/tree/main/supplemental_files)
2. **Python** (3.10–3.12):
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python scripts\00_prepare.py
   ```
3. **Plugin**: apri `plugin` in IntelliJ, task Gradle `build`, copia
   `build/libs/mosca-*.jar` nella cartella `plugins` del server Paper.

## Uso

```
python scripts\02_server.py          (cervelli + esploratore su http://127.0.0.1:8766)
```

In gioco: `/mosca spawn`, `/mosca status`, `/mosca debug`, `/mosca remove [all]`.

Senza comandi: riempi una compostiera **solo di frutta** (mele, anguria, zucca, bacche) e invece
della farina d'ossa, dopo un po', ne esce una mosca.

Ogni player può avere una mosca (`spawn.max-per-player` in `config.yml`); in più il
server dei cervelli rifiuta nuove mosche se il tempo reale medio scende sotto `--spawn-rt`.

## Crediti e licenze

Dati del connettoma: FlyWire Consortium (Dorkenwald et al., Nature 2024; Schlegel et al.,
Nature 2024), licenza **CC BY-NC 4.0**: uso non commerciale con attribuzione.
Modello di base: Shiu et al., "A Drosophila computational brain model reveals sensorimotor
processing", Nature 2024.
