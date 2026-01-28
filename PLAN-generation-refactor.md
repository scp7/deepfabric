# Generation Model Refactor Plan

## Overview

Simplify dataset generation from step-based batching to UUID-based iteration with concurrency control.

## Current Model

```
num_steps = total_samples / batch_size
for step in range(num_steps):
    start_idx = step * batch_size
    prompts = topic_paths[start_idx:start_idx+batch_size]
    process_batch(prompts)
```

- Progress: "Step 3/50"
- Checkpoint tracks `_processed_ids` (topic_ids, may have duplicates in graphs)
- Cycles topics by multiplying path list
- Filtering by `topic_id` causes issues with duplicate UUIDs

## New Model

```
unique_topics = get_unique_topics_by_uuid()  # deduplicated
cycles_needed = ceil(total_samples / len(unique_topics))

for cycle in range(cycles_needed):
    for topic in unique_topics:
        if (topic.uuid, cycle) not in completed:
            await generate_sample(topic)
            mark_completed(topic.uuid, cycle)
```

- Progress: "1234/5000 samples (Cycle 2/3)"
- Checkpoint tracks `{uuid: cycles_completed}` or `set((uuid, cycle))`
- No path-based iteration
- `batch_size` becomes `concurrency` (parallel LLM calls)

---

## Impact Analysis

### 1. Topic Models

#### Graph (deepfabric/graph.py)
- **Has UUIDs**: Yes, each node has `metadata.uuid`
- **Change needed**: Add `get_unique_topics()` method returning deduplicated list by UUID
- **Impact**: Low - structure already supports this

#### Tree (deepfabric/tree.py)
- **Has UUIDs**: No - currently hashes path content
- **Change needed**:
  - Option A: Generate and persist UUIDs when building tree
  - Option B: Add UUID to leaf nodes in JSONL format
- **Impact**: Medium - requires format change for new trees, migration for existing

**Decision needed**: How to handle existing trees without UUIDs?

### 2. Generator (deepfabric/generator.py)

#### Remove
- `num_steps` parameter and calculation
- `_prepare_topic_paths()` cycling logic (path multiplication)
- Step-based loop (`for step in range(num_steps)`)
- `_generate_batch_prompts()` with start_idx
- Step events (`step_start`, `step_complete`)

#### Add
- `concurrency` parameter (rename from `batch_size`)
- `get_unique_topics()` call on topic model
- Cycle-aware iteration
- Semaphore-based concurrency control
- New progress events (`sample_complete`, `cycle_complete`)

#### Modify
- `_processed_ids` → `_completed` as `dict[str, int]` (uuid → cycles_completed)
- Checkpoint format to store completion state
- `_save_checkpoint()` to track by UUID + cycle
- `load_checkpoint()` to restore UUID completion state

### 3. CLI (deepfabric/cli.py)

#### `deepfabric validate`
Current output:
```
Total tree paths available: 2750
Total requested paths: 5000
```

New output:
```
Unique topics (by UUID): 1875
Requested samples: 5000
Cycles needed: 3 (1875 × 3 = 5625, generating 5000)
```

#### `deepfabric start`
- Rename `--batch-size` to `--concurrency` (keep alias for backward compat)
- Remove step-related options if any

### 4. TUI (deepfabric/tui.py)

#### TUI Rich Mode

Current:
```
Step 3/50 [████████░░░░░░░░] 300/5000 samples
```

New:
```
Cycle 2/3 [████████░░░░░░░░] 2100/5000 samples
Topics: 1875 unique │ Concurrency: 10
```

Changes:
- Replace step progress bar with cycle-aware progress
- Show unique topics count in status panel
- Update live table to show cycle info instead of step info
- Update completion summary with cycle stats

#### TUI Simple Mode

Current:
```
Step 1: +100 samples (100/5000)
Step 2: +100 samples (200/5000)
...
```

New:
```
Starting generation: 5000 samples from 1875 unique topics (3 cycles)
Cycle 1: 1875/1875 topics ✓
Cycle 2: 1875/1875 topics ✓
Cycle 3: 1250/1250 topics ✓ (partial)
Complete: 5000 samples generated
```

Changes:
- Print cycle progress instead of step progress
- Show partial cycle indicator
- Periodic sample count updates within cycle (every N samples or every few seconds)

#### Events to Update
- Remove: `step_start`, `step_complete`
- Add: `cycle_start`, `cycle_complete`
- Modify: `generation_start` to include `unique_topics`, `cycles_needed`, `final_cycle_size`
- Modify: `generation_complete` to include cycle stats

### 5. Dataset Manager (deepfabric/dataset_manager.py)

- Update event handlers for new event types
- Update `handle_dataset_events_async()` to handle cycle-based progress

### 6. Checkpoint Format

Current (`_checkpoint_metadata.json`):
```json
{
  "version": "1.0",
  "processed_ids": ["uuid1", "uuid2", ...],
  "total_samples": 500,
  "total_failures": 10
}
```

New:
```json
{
  "version": "2.0",
  "completed": [
    ["uuid1", 0],
    ["uuid1", 1],
    ["uuid2", 0],
    ...
  ],
  "total_samples": 500,
  "total_failures": 10,
  "cycles_needed": 3,
  "unique_topics": 1875
}
```

Where `completed` is a list of `[uuid, cycle]` tuples (JSON arrays).
In Python: `self._completed: set[tuple[str, int]]`

### 7. Tree JSONL Format

Current:
```json
{"path": ["Root", "Branch", "Leaf"]}
```

New:
```json
{"path": ["Root", "Branch", "Leaf"], "leaf_uuid": "550e8400-e29b-41d4-a716-446655440000"}
```

- UUID generated at build time using `uuid.uuid4()`
- Persisted with tree, stable across loads
- Existing trees without `leaf_uuid` are incompatible (require rebuild)

---

## Migration & Compatibility

### Existing Trees
- Trees without `leaf_uuid` field are incompatible
- On load, detect missing UUIDs and raise error with clear message:
  ```
  Error: Tree format outdated. Missing leaf_uuid fields.
  Please rebuild your topic tree with: deepfabric topics build ...
  ```

### Existing Checkpoints
- Version 1.0 checkpoints are incompatible
- On resume attempt, detect version and refuse:
  ```
  Error: Checkpoint format v1.0 is incompatible with current version.
  Please delete checkpoint and restart: rm -rf .checkpoints/
  ```

### Backward Compatibility
- CLI: `--batch-size` kept as alias for `--concurrency` (deprecated warning)
- YAML config: `batch_size` still accepted, mapped to `concurrency` internally
  ```yaml
  # Old (still works, no warning in config)
  data_engine:
    args:
      batch_size: 100

  # New (preferred)
  data_engine:
    args:
      concurrency: 100
  ```
- API: `batch_size` parameter still accepted, mapped to `concurrency`

---

## Implementation Order

### Phase 1: Topic Model Updates
1. Add `get_unique_topics()` to `TopicModel` base class (returns deduplicated by UUID)
2. Implement in `Graph` - deduplicate by node UUID from metadata
3. Update `Tree` to generate and persist `leaf_uuid` on build
4. Update `Tree.save()` to include `leaf_uuid` in JSONL
5. Update `Tree.from_dict_list()` to load `leaf_uuid` (error if missing)
6. Implement `get_unique_topics()` in `Tree`

### Phase 2: Generator Core
7. Define new checkpoint format v2.0 with `(uuid, cycle)` tuples
8. Update `load_checkpoint()` to detect v1 and reject with clear error
9. Replace step loop with cycle-based UUID iteration
10. Change `_processed_ids: set[str]` → `_completed: set[tuple[str, int]]`
11. Implement concurrency with `asyncio.Semaphore` (replace batch processing)
12. Update `_save_checkpoint()` for new format
13. Remove `_prepare_topic_paths()`, `_generate_batch_prompts()`, step logic

### Phase 3: CLI & Validation
14. Update `validate` command: show unique UUIDs, cycles needed
15. Add `--concurrency` option, keep `--batch-size` as deprecated alias (CLI only)
16. Update config loader: accept both `batch_size` and `concurrency` in YAML (no deprecation warning for config)
17. Update help text and examples

### Phase 4: TUI Rich Mode
17. Update `generation_start` handler - show unique topics, cycles
18. Replace step progress with cycle progress bar
19. Update live stats panel
20. Update completion summary with cycle stats

### Phase 5: TUI Simple Mode
21. Update `generation_start` - print unique topics, cycles info
22. Replace step output with cycle output
23. Add periodic progress within cycle (every N samples)
24. Update completion message

### Phase 6: Event System
25. Remove events: `step_start`, `step_complete`
26. Add events: `cycle_start`, `cycle_complete`
27. Update `generation_start` event payload
28. Update `generation_complete` event payload
29. Update `dataset_manager.py` event handlers

### Phase 7: Testing & Cleanup
30. Update unit tests for new topic model methods
31. Update generator tests for cycle-based logic
32. Update TUI tests
33. Add migration/compatibility tests (v1 checkpoint rejection, old tree rejection)
34. Remove dead code (step-related functions)
35. Update documentation

---

## Decisions

1. **Tree UUIDs**: Generate on build and persist in JSONL
   - New format: `{"path": [...], "leaf_uuid": "..."}`
   - Existing trees require regeneration

2. **Checkpoint migration**: Require fresh start
   - Detect v1 checkpoint, warn user, refuse to resume
   - User must delete checkpoint or start fresh

3. **Partial cycles**: Track as `(uuid, cycle)` tuples
   - Example: 5000 samples from 1875 topics = 3 cycles
   - Cycle 1: 1875 topics, Cycle 2: 1875 topics, Cycle 3: 1250 topics (partial)
   - Checkpoint stores: `{("uuid-abc", 0), ("uuid-abc", 1), ("uuid-def", 0), ...}`

4. **Progress granularity**: Batch updates
   - Update TUI every N samples or every second (reduce overhead)
   - Rich mode: live updating progress bar
   - Simple mode: periodic line output
