"""Clean test pollution from real HistoryStore."""
import json
from pathlib import Path
from anylabeling.services.training_center.history import get_history_store

store = get_history_store()
all_jobs = store.list_jobs()

# Test display names to remove
test_names = {
    'Test Job', 'New', 'Second Job', 'Invalid Test', 'Long Test',
    'Invalid Python Test', 'Failure Test', 'Qt Test', 'Invalid Script Test',
    'E2E Test', 'Arguments Test', 'QSignalSpy Test', 'Repeat Stop', 'Silent',
    'Stderr Only', 'Stdout Only', 'Quick Test', 'Fail Test', 'Normal Test',
}

# Test job_id patterns  
test_prefixes = (
    'test-job-', 'qt-test-', 'test-e2e-', 'qspy-test-',
)

# Test fixture job IDs
test_job_ids = {'custom-001', 'guided-001'}

keep = []
remove = []
for job in all_jobs:
    name = job.display_name or ''
    jid = job.job_id or ''

    if name in test_names:
        remove.append(job)
        continue
    if jid.startswith(test_prefixes):
        remove.append(job)
        continue
    if jid in test_job_ids:
        remove.append(job)
        continue
    ws = job.workspace or ''
    if 'pytest-of-' in ws or 'Temp\\pytest-' in ws:
        remove.append(job)
        continue
    keep.append(job)

print(f"Keep: {len(keep)}")
for j in keep:
    print(f"  KEEP: {j.job_id[:30]:30s} | {j.status:12s} | {j.display_name or ''}")
print(f"\nRemove: {len(remove)}")
for j in remove:
    print(f"  REMOVE: {j.job_id[:30]:30s} | {j.status:12s} | {j.display_name or ''}")
print(f"\nTotal before: {len(all_jobs)}, Keep: {len(keep)}, Remove: {len(remove)}")

# Actually remove from index.jsonl
if remove:
    remove_ids = {j.job_id for j in remove}
    index_file = store.index_file
    lines = []
    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get('job_id') in remove_ids:
                    continue
                lines.append(line)
            except json.JSONDecodeError:
                lines.append(line)
    
    with open(index_file, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    
    # Clear cache and reload
    store._loaded = False
    store._cache.clear()
    store._ensure_loaded()
    
    remaining = store.list_jobs()
    print(f"\nAfter cleanup: {len(remaining)} records")
    for j in remaining:
        print(f"  {j.job_id[:30]:30s} | {j.status:12s} | {j.display_name or ''}")
