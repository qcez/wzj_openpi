#!/usr/bin/env python3
"""验证合并前后：某 episode 某帧 的 `task_index` 指向的 task 文本是否一致。

用法示例：
  python3 scripts/verify_task_mapping.py \
    --orig-repo /path/to/orig_repo \
    --orig-episode 12 \
    --frame-row 0 \
    --merged-repo /path/to/merged_repo \
    --merged-episode 12

如果合并时 episode index 发生了变化，请在 `--merged-episode` 指定新索引。
"""
from pathlib import Path
import argparse
import json
import glob
import sys

try:
    import pyarrow.parquet as pq
except Exception:
    print('ERROR: pyarrow required. Install with pip install pyarrow')
    raise


def find_episode_parquet(repo_path, episode_index):
    repo = Path(repo_path)
    pattern = str(repo / '**' / f'episode_{int(episode_index):06d}.parquet')
    matches = glob.glob(pattern, recursive=True)
    return Path(matches[0]) if matches else None


def read_task_text_from_tasks(tasks_path, idx):
    if not tasks_path.exists():
        return None
    with open(tasks_path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ti = obj.get('task_index')
            if ti is None:
                # if tasks.jsonl is just sequential without explicit indices, use line count
                pass
            if ti == idx:
                return obj.get('task') or obj.get('text')
    # fallback: attempt to index by line number if explicit indices missing
    with open(tasks_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == idx:
                try:
                    obj = json.loads(line.strip())
                    return obj.get('task') or obj.get('text')
                except Exception:
                    return line.strip()
    return None


def load_task_index_from_parquet(pq_path, frame_row):
    # Try pyarrow first, then fall back to pandas
    reader = None
    vals = None
    try:
        tbl = pq.read_table(str(pq_path), columns=['task_index'])
        col = tbl.column('task_index')
        vals = col.to_pylist()
        reader = 'pyarrow'
    except Exception as e_py:
        try:
            import pandas as _pd
            df = _pd.read_parquet(str(pq_path))
            if 'task_index' not in df.columns:
                raise RuntimeError('task_index column missing in parquet')
            vals = df['task_index'].tolist()
            reader = 'pandas'
        except Exception as e_pd:
            raise RuntimeError(f'Failed to read parquet with pyarrow ({e_py}) and pandas ({e_pd})')
    if frame_row < 0 or frame_row >= len(vals):
        raise IndexError(f'frame_row {frame_row} out of range (0..{len(vals)-1})')
    # print which reader was used for debugging
    # (kept as stdout so caller can see it)
    print(f'Read parquet {pq_path} with {reader}, rows={len(vals)}')
    return int(vals[frame_row])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--orig-repo', required=True)
    p.add_argument('--orig-episode', required=True, type=int)
    p.add_argument('--frame-row', required=True, type=int, help='0-based row in the episode parquet')
    p.add_argument('--merged-repo', required=True)
    p.add_argument('--merged-episode', required=True, type=int)
    return p.parse_args()


def main():
    args = parse_args()

    orig_pq = find_episode_parquet(args.orig_repo, args.orig_episode)
    if not orig_pq:
        print('ERROR: original parquet not found for', args.orig_episode, 'under', args.orig_repo)
        sys.exit(2)

    merged_pq = find_episode_parquet(args.merged_repo, args.merged_episode)
    if not merged_pq:
        print('ERROR: merged parquet not found for', args.merged_episode, 'under', args.merged_repo)
        sys.exit(2)

    print('Using original parquet:', orig_pq)
    print('Using merged parquet:  ', merged_pq)

    orig_task_idx = load_task_index_from_parquet(orig_pq, args.frame_row)
    merged_task_idx = load_task_index_from_parquet(merged_pq, args.frame_row)

    orig_tasks_path = Path(args.orig_repo) / 'meta' / 'tasks.jsonl'
    merged_tasks_path = Path(args.merged_repo) / 'meta' / 'tasks.jsonl'

    orig_task_text = read_task_text_from_tasks(orig_tasks_path, orig_task_idx)
    merged_task_text = read_task_text_from_tasks(merged_tasks_path, merged_task_idx)

    print('\nOriginal:')
    print(' repo:', args.orig_repo)
    print(' episode:', args.orig_episode, 'frame_row:', args.frame_row)
    print(' task_index:', orig_task_idx)
    print(' task_text:', orig_task_text)

    print('\nMerged:')
    print(' repo:', args.merged_repo)
    print(' episode:', args.merged_episode, 'frame_row:', args.frame_row)
    print(' task_index:', merged_task_idx)
    print(' task_text:', merged_task_text)

    ok = (orig_task_text == merged_task_text)
    print('\nMATCH:' , ok)
    if not ok:
        print('Note: texts may differ due to duplicates or tokenization; inspect tasks.jsonl lines around the indices.')


if __name__ == '__main__':
    main()
