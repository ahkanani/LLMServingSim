# Internship release reconstruction

The local branch `codex/bd-release-reconstructed` rebuilds the internship release
on `f6848b8a0077daa05a79fe657d03be9d141893b6`, then merges current main
`a4053bc1161872420e1e0607cb3409ef659b828e`. Nothing has been pushed.

## History and provenance

The published release was `a4053bc -> b7e403d -> b9f2b21`: its snapshot replaced
newer upstream implementations with code developed against an older base.
The supplied base is consistent with the snapshot comparison; original
internship commit history is unavailable, so this reconstructs ancestry rather
than recovering those historical commits.

- `codex/bd-release-original` preserves `b9f2b21` locally.
- `bd-opensource-release` and `main` were left unchanged.
- `2f0ef37` has parent `f6848b8` and a tree **identical** to `b9f2b21`, including
  the historical vendored backend snapshot.
- `6fbd336` merges `a4053bc` into that reconstruction, resolves the frontend
  changes, and restores separate backend repositories.

Current main is an ancestor of the result. Future main changes can therefore
be merged using genuine shared ancestry. Integration still requires resolving
any new overlapping changes.

## Integration decisions

Retained main's block-pool KV cache and unified scheduler, schedule-time token
accounting, pipeline block boundaries, DP FIFOs and dummy-batch fixes, P/D KV
transfer sizing, in-process Chakra conversion, graph caching, and pass/deadline
IPC. The internship's scheduler and request changes relative to the old base
were whitespace-only, so those files use main's implementations.

Retained and adapted expert offloading, detailed/custom routing, configurable
network topology, DeepSeek profiling overrides, H200 profile bundles, model
configs, experiment configs, and the trace-replay project. Expert trace emitters
now supply field tuples to main's in-process converter. The trace-replay tree
and internship H200 bundles are identical to the original release.

Restored main's documentation deployment workflow and recorded example outputs.
The deleted radix-tree cache stays deleted; the current block pool handles KV
storage. Expert-offload usage is documented on the website.

## Separate backend repositories

All changed repositories have local branch `codex/bd-release-reconstructed`:

| Repository path | Review base | Local result |
| --- | --- | --- |
| `astra-sim` | `d346994` | `6fa0a06` |
| `astra-sim/extern/graph_frontend/chakra` | `30221ab` | `d9889e9` |
| `astra-sim/extern/memory_backend/analytical` | `62313ef` | `60476d3` |

ASTRA's release changes relative to its old base `f82fb3d` were whitespace-only.
Its final C++ implementation matches main's backend; its only differences from
`d346994` are the Chakra and analytical-memory submodule references.

Chakra combines main's pipeline and in-process conversion changes with the
internship's expert-PIM fork/join dependencies. Analytical memory skips memory
service time for zero-byte PIM compute/transfer nodes, whose runtime is already
explicit in the trace. The original nested `.gitmodules` structure is retained;
helper and network backends remain separate repositories as well.

These submodule commits are **local only**. A remote clone cannot retrieve them
until the dependent repositories are published. For a later publication, the
memory and Chakra commits must be available before ASTRA, and ASTRA before the
parent repository. Publication destinations can be chosen during review.

## Validation

Built both ASTRA analytical binaries from source and installed the merged Chakra
converter into an isolated environment at `/tmp/bd-release-venv`.

The final validation passes **58 of 58** clock baselines after the two
intentional baseline updates below. The first full validation reproduced
56 of 58 clock baselines and all four
benchmark examples' `sim.csv` and `summary.txt` digests. The two deliberate
behavior changes already present in the internship code were independently
checked, then their baselines were updated:

| Scenario | Main baseline (ns) | Integrated result (ns) | Reason |
| --- | --- | --- | --- |
| `expert_routing_rr` | 1429681327 | 1625988669 | RR advances token indices instead of routing every token with index zero. |
| `no_block_copy` | 1037528119 | 1945309759 | With replay disabled, all transformer blocks are emitted; main emitted only one. |

Direct emitted-trace checks verify all 48 transformer layers for BALANCED, RR,
and RAND with replay disabled, and conservation of RR expert-token pairs.
The Qwen3-235B FP8 P/D expert-offload example completed on ten requests at
2030929786 ns. Its converted graph contains 94 verified hot/cold fork/join
structures: hot computation and input transfer share the dispatch dependency,
output transfer waits for cold PIM compute, and the join waits for both paths.
The example uses the supplied **synthetic** PIM profile; this checks integration,
not hardware performance accuracy.

Built a small synthetic TracePack and ran the trace-replay simulator through
metrics output. Python source compilation and the simulator/replay CLI checks
passed. The Docusaurus production build and rendered-markup checks passed
(62 pages and 21 Mermaid diagrams), using Node 22 and pnpm 10.

Commands used:

```bash
bash astra-sim/build/astra_analytical/build.sh
PYTHON=/tmp/bd-release-venv/bin/python TIMEOUT=120 bash serving/validate.sh
PYTHON=/tmp/bd-release-venv/bin/python bash serving/validate.sh --update expert_routing_rr no_block_copy
/tmp/bd-release-venv/bin/python -m serving \
  --cluster-config configs/cluster/pim_experiments/expert_pim/01_offload8_equal.json \
  --dataset workloads/example_trace.jsonl --num-reqs 10 \
  --expert-routing-policy CUSTOM --log-level WARNING --save-trace-text \
  --run-id bd_offload_check
npx --yes --package=node@22 --package=pnpm@10 \
  -c 'cd docs && pnpm install --frozen-lockfile && pnpm build'
```

GPU/vLLM profiling was not rerun, and the ns-3 and HTSim binaries were not built.
External full-size trace datasets were not required for the synthetic replay
check. Generated validation logs and plots are excluded from the final changes.

## Local review and recovery

Inspect the integrated changes with:

```bash
git diff main..codex/bd-release-reconstructed
git log --graph --oneline main..codex/bd-release-reconstructed
git -C astra-sim diff d346994..codex/bd-release-reconstructed
git -C astra-sim/extern/graph_frontend/chakra diff 30221ab..codex/bd-release-reconstructed
git -C astra-sim/extern/memory_backend/analytical diff 62313ef..codex/bd-release-reconstructed
```

Verified incremental Git bundles and validation evidence are saved under
`$(git rev-parse --git-common-dir)/bd-release-review/`, outside this worktree.
Each bundle includes the corresponding local branch and requires the review
base in the table above (parent repository: `a4053bc`). Restore into a clone
that contains its base using:

```bash
git fetch /path/to/repository.bundle \
  refs/heads/codex/bd-release-reconstructed:refs/heads/codex/bd-release-reconstructed
```

Use the matching bundle separately in each repository. The review directory
also contains `feature-check.py`, its output, and build/validation logs.
