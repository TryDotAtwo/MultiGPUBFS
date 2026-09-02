# REF-046: discovery-publication interleavings — not run

Date: 2026-08-28

Historical failed-attempt record. Superseded as the overall REF-046 status by
the [completed bounded-model report](REF-046-discovery-publication-interleavings.md)
dated 2026-08-29. The observations below describe the unavailable environment
at those attempts, not the current Docker state or the final experiment result.

## Question

Do bounded interleavings distinguish blind already-visited drop from a helpable
claim descriptor and a logged publication intent under injected worker stops?

## Intended scope

- Rust-only finite state model, built and executed in Docker.
- Test-first cases for the path `s->a->b->c`.
- Exhaustive declared schedules over claim, publish, retry/help, and stop events.
- Separate assertions for reachability coverage, outstanding obligations,
  duplicate physical publication, and termination.
- No timing, optimizer, GPU code, or production protocol.

## Readiness observations

An initial

```text
docker ps --format ...
```

returned no output and was initially interpreted as an empty healthy daemon.
That inference was invalid: it did not provide positive server identity.
The next server-dependent query returned:

```text
permission denied while trying to connect to the docker API at
npipe:////./pipe/dockerDesktopLinuxEngine
```

The authoritative check

```text
docker info --format "{{.ServerVersion}}"
```

then returned exit code 1 with the same permission error.

### 2026-08-29 recheck

The environment was checked again before attempting the preserved fixture:

```text
docker version --format '{{json .Server}}'
docker ps --format '{{.ID}} {{.Image}} {{.Status}}'
```

The server value was `null`, and both server-dependent accesses reported:

```text
permission denied while trying to connect to the docker API at
npipe:////./pipe/dockerDesktopLinuxEngine
```

At that recheck the fixture remained unrun. The check confirmed Docker
unavailability for that attempt; it added no BFS or protocol result.

## Status

Not run at the recorded attempts. No Rust test or implementation file was
written in those attempts because the required Docker RED step could not
execute. No infrastructure repair was attempted. Later implementation and
finite results are recorded separately in the linked completed report.

## Preserved unknowns

- Whether the proposed finite event alphabet covers every semantic gap needed
  for the three protocol classes.
- Which schedules leave blind drop with visited membership but no expansion
  duty.
- Whether the helpable and logged models preserve responsibility under every
  declared single-stop schedule.
- Which duplicate-publication schedules remain safe for set output but fail for
  contribution-count output.

This artifact is infrastructure evidence only. It supplies no algorithmic
result for the note 178 contract.
