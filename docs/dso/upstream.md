# Upstream AI2-THOR Snapshot

## Recorded Source

- Local source commit or snapshot commit: `f2021760f33cf295bcaafb97f589111208648068`
- Official upstream repository: <https://github.com/allenai/ai2thor>
- Expected upstream remote name: `upstream`
- Unity editor version: `2020.3.25f1`
- Unity editor revision: `2020.3.25f1 (9b9180224418)`

The local repository currently has one Git commit, so it should be treated as a source snapshot rather than a clone with full upstream AI2-THOR history.

## Remote Setup

Git remote configuration is local machine state and is not shared through the repository. Collaborators can add the upstream remote with:

```bash
git remote add upstream https://github.com/allenai/ai2thor.git
git fetch upstream
```

If `upstream` already exists, verify it with:

```bash
git remote -v
```

## Future Upstream Changes

Future upstream changes should be incorporated deliberately. Before importing upstream changes, record the target upstream commit, review possible conflicts with DSO-owned files, run the relevant lightweight checks, and document any required manual resolution in the pull request.
