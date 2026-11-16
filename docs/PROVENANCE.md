# Provenance: the public commits, and how a gate result proves its TOML

## What the public repository is

The repository is developed in the pod's internal git (the Buzz git host, `docs/POD-GIT.md`): one branch per kanban
card, merged to `main` on review. The public mirror is a **scrubbed export** of that `main`, made from the laptop
session: hostnames, network addresses and home paths are rewritten, gate TOMLs included, and the export is
committed as new history. So:

- The public history is short: it holds export commits, not the commits the work was done in.
- Card ids (10 hex digits, e.g. `card/3bed1267a2`) and commit shas cited in gate files, summaries, manifests and
  `configs/champions.toml` refer to the internal history. They identify the work on the pod's board and in the
  internal repository; they do not resolve in the public one.
- Paths such as `/data/...` or `/mnt/...` in results name places on the pod; nothing in this tree needs them to
  exist.

## What the export removes, and how to check it

The export copies `main`, then a scrub script rewrites text files. Vendored source is
`catalog/<origin other than flerp>/*/*/src/`; it keeps its upstream bytes, except that StrixFold (the pod's own
fork) gets the address and key rewrites below and nothing else. Files that do not fit that layout, such as
`catalog/strixfold_pr6/*/overlay/zig/src/*.patch`, are treated as first-party and get every rewrite.

- each pod network prefix becomes an RFC 5737 documentation range (`192.0.2.0/24`, `198.51.100.0/24`,
  `203.0.113.0/24`), whatever follows the prefix, so an address and a bare prefix in the same gate map alike;
- home directories become `/home/user` and `/home/pod`; host, user and agent names become `control-node`,
  `relay-host`, `pod`, `owner` and `agent`; the relay owner's public key becomes `<relay-owner-pubkey>`.

The scrub script and its deny list live with the maintainer on the relay host, outside this repository, because
their rules spell out the very names they remove. What anyone can run is the check the export must pass before it
publishes:

```
python3 tools/public-check.py [ROOT] [--deny FILE] [--list]   # exit 0 = no private address, home directory or deny-list word
```

On the internal tree it finds many hits; on the public tree it finds none (the export runs it with its deny list).
Gate TOMLs are scrubbed too, which is why their public sha256 differs (next section).

## How a gate result proves which TOML it ran under

A gate is written down before it runs: its TOML (`gate/<name>.toml`, `gate/interface/<name>.toml`) carries the
criteria, cases and pass rules. A result proves which TOML judged it with a `config_sha256` field, the sha256 of
the TOML's bytes at run time, in its JSON summary (`result.json`, `summary.json`, `verdict.json`, ...). The shared
summary helper the gate runners write through records it; results written before it, and runners that do not use
it, record nothing.

On the pod a recorded sha is the sha of the internal TOML. The public export rewrites some TOMLs, so their public
bytes have a different sha. The export therefore writes `gate/PROVENANCE-MAP.tsv`, one row per gate TOML:

```
toml_path<TAB>internal_sha256<TAB>public_sha256
```

made by `tools/provenance-map.py --internal <internal tree> --public <public tree>`. The internal tree has no map.

## Checking it: `tools/gate-provenance.py`

```
python3 tools/gate-provenance.py            # table of every result; exit 0
python3 tools/gate-provenance.py --json     # the same as JSON
python3 tools/gate-provenance.py --strict   # exit 1 if any result is unrecorded, unmatched or has no gate TOML
```

A result is each directory under `gate/results/` (each `gate/results/interface/<name>/` separately) or a file
directly under it. Its gate TOML is the one its JSON names (`config`, `config_path`, `criteria_file`), else the TOML
of the same name, else the gate TOML whose name is the longest prefix of the result's. Each recorded
`config_sha256` is reported as:

| state | meaning |
|---|---|
| `current` | equals the sha256 of the TOML's bytes in this tree |
| `current-internal` | equals the map's `internal_sha256` for that TOML: the result ran under the internal bytes the public file was exported from (public tree only) |
| `older <commit>` | equals the TOML's bytes at an earlier commit of this repository (the oldest commit with those bytes is named); in the public tree only the export history is searched |
| `unmatched` | none of these: the TOML changed in history this tree does not hold, or the result was not judged under it |
| `unrecorded` | the result records no `config_sha256`; listed, not a failure |

The direct match is tried first, then the map, then git history.
