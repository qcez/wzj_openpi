#!/usr/bin/env python3
"""
完整合并多个 LeRobot 子仓库到一个仓库的新实现。

功能要点：
- 按输入仓库顺序直接拼接 `meta/tasks.jsonl`（不去重），并为每个子仓库计算 task_index 偏移量；
- 遍历每个子仓库的 `meta/episodes.jsonl`，读取对应 episode parquet，重映射 `task_index`（加偏移）、确保 `frame_index` 存在、替换 `episode_index` 为全局新 id、重写全局 `index`（连续计数）；
- 按 `--chunk_size` 将 episode 写入 `data/chunk-XXX/episode_{:06d}.parquet`（每 chunk 包含 chunk_size 个 episode）；
- 生成完整的 `meta/episodes.jsonl`、`meta/episodes_stats.jsonl`，并写入修正后的 `meta/info.json`（包括 `splits.train` = "0:total_parquets"）。

注意：本脚本依赖 `pyarrow`，并在遇到无法读取的 parquet 文件时记录错误并中止，避免静默产生不一致数据。
"""
from pathlib import Path
import argparse
import json
import os
import glob
import shutil
import sys

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except Exception:
    print("ERROR: pyarrow is required. Install with `pip install pyarrow`.")
    raise


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path, objs):
    with open(path, 'w', encoding='utf-8') as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def find_episode_parquet_in_repo(repo_data_dir, episode_index):
    pattern = str(Path(repo_data_dir) / '**' / f'episode_{int(episode_index):06d}.parquet')
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def merge_repos(input_repos, output_repo, chunk_size=1000, dry_run=False, auto_continue=False):
    output_repo = Path(output_repo)
    meta_out = output_repo / 'meta'
    data_out = output_repo / 'data'
    ensure_dir(meta_out)
    ensure_dir(data_out)

    total_tasks = 0
    # collect per-repo original task entries and global ordered list
    repo_task_entries = {}  # repo_path -> list of (orig_idx, task_text)
    global_task_entries = []  # list of (repo_path, orig_idx, task_text)
    repo_task_new_index_map = {}  # repo_path -> {orig_idx: new_idx}

    # 1) concatenate tasks.jsonl in repo order, record per-repo offset
    for repo_path in input_repos:
        repo = Path(repo_path)
        meta_dir = repo / 'meta'
        tasks_path = meta_dir / 'tasks.jsonl'
        repo_key = str(repo)
        repo_task_entries[repo_key] = []
        if not tasks_path.exists():
            print(f'WARN: tasks.jsonl not found in {repo}; skipping')
            repo_task_new_index_map[repo_key] = {}
            continue
        for line in open(tasks_path, 'r', encoding='utf-8'):
            line=line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # fallback: treat whole line as task string
                obj = {'task': line}
            orig_idx = obj.get('task_index')
            task_text = obj.get('task') if 'task' in obj else obj.get('text') if 'text' in obj else None
            repo_task_entries[repo_key].append((orig_idx, task_text))
            global_task_entries.append((repo_key, orig_idx, task_text))
            total_tasks += 1

    # assign new sequential task_index by concatenating per-repo entries in input_repos order
    new_tasks_lines = []
    repo_task_new_index_map = {}
    cumulative = 0
    for repo_path in input_repos:
        repo_key = str(Path(repo_path))
        entries = repo_task_entries.get(repo_key, [])
        for (orig_idx, task_text) in entries:
            new_tasks_lines.append({'task_index': cumulative, 'task': task_text})
            repo_task_new_index_map.setdefault(repo_key, {})[orig_idx] = cumulative
            cumulative += 1

    total_tasks = cumulative

    if dry_run:
        print('DRY RUN: would write total tasks:', total_tasks)
    else:
        with open(meta_out / 'tasks.jsonl', 'w', encoding='utf-8') as f:
            for obj in new_tasks_lines:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print('Wrote meta/tasks.jsonl with', total_tasks, 'entries')

    # 2) iterate episodes and rewrite parquets
    new_episodes = []
    new_episodes_stats = []
    current_global_index = 0
    new_episode_idx = 0
    failures = []
    missing_parquet_count = 0
    unmatched_stats_count = 0
    unmatched_stats_examples = []

    for repo_path in input_repos:
        repo = Path(repo_path)
        meta_dir = repo / 'meta'
        data_dir = repo / 'data'
        repo_key = str(repo)

        episodes_path = meta_dir / 'episodes.jsonl'
        if not episodes_path.exists():
            print(f'WARN: episodes.jsonl not found in {repo}; skipping')
            continue

        # load repo's episodes_stats list (preserve original order)
        repo_episodes_stats_list = []
        episodes_stats_path = meta_dir / 'episodes_stats.jsonl'
        if episodes_stats_path.exists():
            for s in read_jsonl(episodes_stats_path):
                repo_episodes_stats_list.append(s)

        # mapping from this repo's old episode_index -> new global episode_index
        repo_episode_index_map = {}
        for ep in read_jsonl(episodes_path):
            # locate parquet
            src_pq = None
            if 'data_path' in ep and ep['data_path']:
                # data_path may be a template; try join
                candidate = repo / ep['data_path']
                if candidate.exists():
                    # if it's a file, use it; if dir, search
                    if candidate.is_file():
                        src_pq = str(candidate)
                    else:
                        # search for episode_{:06d}.parquet
                        src_pq = find_episode_parquet_in_repo(candidate, ep.get('episode_index'))
            if src_pq is None:
                # fallback: search under repo/data
                src_pq = find_episode_parquet_in_repo(repo / 'data', ep.get('episode_index'))

            if not src_pq:
                failures.append((str(repo), ep.get('episode_index'), 'parquet_not_found'))
                missing_parquet_count += 1
                print(f'ERROR: parquet for episode {ep.get("episode_index")} not found in {repo}')
                if not auto_continue:
                    raise RuntimeError('Missing parquet; aborting')
                else:
                    continue

            # read parquet
            try:
                pf = pq.ParquetFile(src_pq)
                table = pf.read()
            except Exception as e:
                failures.append((src_pq, 'read_error', str(e)))
                print(f'ERROR: failed to read parquet {src_pq}: {e}')
                if not auto_continue:
                    raise
                else:
                    continue

            # remap task_index column if present using per-repo mapping
            cols = table.column_names
            if 'task_index' in cols:
                arr = table.column('task_index').to_pandas()
                import numpy as np
                repo_key = str(repo)
                mapping = repo_task_new_index_map.get(repo_key, {})
                # vectorized mapping
                new_task_idxs = np.array([mapping.get(int(x), -1) for x in arr])
                if (new_task_idxs == -1).any():
                    bad_idx = set(int(x) for i,x in enumerate(arr) if new_task_idxs[i] == -1)
                    failures.append((src_pq, 'task_index_map_missing', list(bad_idx)))
                    print(f'ERROR: missing task_index mapping for repo {repo} values {list(bad_idx)}')
                    if not auto_continue:
                        raise RuntimeError('Missing task_index mapping; aborting')
                # replace column
                table = table.remove_column(table.column_names.index('task_index'))
                table = table.append_column('task_index', pa.array(new_task_idxs))

            # ensure frame_index
            if 'frame_index' not in table.column_names:
                n = table.num_rows
                table = table.append_column('frame_index', pa.array(list(range(n))))

            # overwrite episode_index with new_episode_idx
            if 'episode_index' in table.column_names:
                table = table.remove_column(table.column_names.index('episode_index'))
            table = table.append_column('episode_index', pa.array([new_episode_idx] * table.num_rows))

            # rewrite global index
            if 'index' in table.column_names:
                table = table.remove_column(table.column_names.index('index'))
            new_index_arr = list(range(current_global_index, current_global_index + table.num_rows))
            table = table.append_column('index', pa.array(new_index_arr))

            # write to output chunk dir (but first verify index/episode_index columns)
            chunk_id = new_episode_idx // int(chunk_size)
            chunk_dir = data_out / f'chunk-{chunk_id:03d}'
            ensure_dir(chunk_dir)
            out_pq = chunk_dir / f'episode_{new_episode_idx:06d}.parquet'

            # verification: ensure episode_index and index columns were set as expected
            try:
                epi_check = table.column('episode_index').to_pylist()
                if any(int(x) != int(new_episode_idx) for x in epi_check):
                    raise ValueError(f'episode_index column not all {new_episode_idx}: {epi_check[:5]}')
                idx_check = table.column('index').to_pylist()
                expected_start = current_global_index
                if not (int(idx_check[0]) == expected_start and int(idx_check[-1]) == expected_start + table.num_rows - 1):
                    raise ValueError(f'index column not continuous starting at {expected_start}: start={idx_check[0]}, end={idx_check[-1]}')
            except Exception as e:
                print('ERROR: verification failed before writing parquet for episode', ep.get('episode_index'), ':', e)
                failures.append((str(src_pq), 'verification_failed', str(e)))
                if not auto_continue:
                    raise
                else:
                    # skip writing this episode
                    continue

            if dry_run:
                print('DRY RUN: would write', out_pq)
            else:
                try:
                    pq.write_table(table, str(out_pq))
                except Exception as e:
                    failures.append((str(out_pq), 'write_error', str(e)))
                    print(f'ERROR: failed to write parquet {out_pq}: {e}')
                    if not auto_continue:
                        raise
                    else:
                        continue

            # record episode meta
            ep_out = dict(ep)
            # set episode_index to new (do NOT include data_path or n_frames — training uses info.data_path template)
            old_ep_idx = ep.get('episode_index')
            ep_out['episode_index'] = new_episode_idx
            if 'data_path' in ep_out:
                del ep_out['data_path']
            if 'n_frames' in ep_out:
                del ep_out['n_frames']
            # remember mapping
            repo_episode_index_map[old_ep_idx] = new_episode_idx
            new_episodes.append(ep_out)

            # Build episode stat by trying to find the original repo's episodes_stats entry
            orig_stat = None
            for s in repo_episodes_stats_list:
                if s.get('episode_index') == old_ep_idx:
                    orig_stat = s
                    break
            if orig_stat:
                stat = dict(orig_stat)
            else:
                stat = dict(ep)
            # update indices; remove data_path so training uses info template; keep n_frames
            stat['episode_index'] = new_episode_idx
            if 'data_path' in stat:
                del stat['data_path']
            n_frames = table.num_rows
            # do not write n_frames into episodes_stats.jsonl per user request
            new_episodes_stats.append(stat)

            # advance counters
            new_episode_idx += 1
            current_global_index += n_frames

        # After processing episodes for this repo, append any repo episodes_stats entries
        # that were not matched above, preserving original order. For unmatched entries
        # we attach an `orig_repo` field so they are not silently lost.
        for s in repo_episodes_stats_list:
            old_idx = s.get('episode_index')
            if old_idx in repo_episode_index_map:
                # already processed and added
                continue
            # map fields where possible
            s_out = dict(s)
            mapped = repo_episode_index_map.get(old_idx)
            if mapped is not None:
                s_out['episode_index'] = mapped
            else:
                s_out['orig_repo'] = str(repo)
                unmatched_stats_count += 1
                if len(unmatched_stats_examples) < 20:
                    unmatched_stats_examples.append({'repo': str(repo), 'episode_index': old_idx, 'entry': s_out})
            # keep other fields, do not inject n_frames
            new_episodes_stats.append(s_out)

    # 3) write meta files
    if dry_run:
        print('DRY RUN: would write meta/episodes.jsonl with', len(new_episodes))
        print('DRY RUN: would write meta/episodes_stats.jsonl with', len(new_episodes_stats))
        print('DRY RUN: would write meta/info.json with totals and splits')
        return

    write_jsonl(meta_out / 'episodes.jsonl', new_episodes)
    write_jsonl(meta_out / 'episodes_stats.jsonl', new_episodes_stats)
    print('Wrote episodes and episodes_stats')

    # compute total_videos under output_repo/videos (real count)
    video_count = 0
    videos_root = output_repo / 'videos'
    if videos_root.exists():
        for p in videos_root.rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.mp4', '.mkv', '.avi', '.mov', '.webm'):
                video_count += 1

    # write info.json: preserve structure from first input repo if available
    info = {}
    if input_repos:
        first_info = Path(input_repos[0]) / 'meta' / 'info.json'
        if first_info.exists():
            try:
                with open(first_info, 'r', encoding='utf-8') as f:
                    info = json.load(f)
            except Exception:
                info = {}

    # update canonical counts
    info['total_episodes'] = len(new_episodes)
    info['total_frames'] = current_global_index
    info['total_tasks'] = total_tasks
    info['total_videos'] = video_count
    info['total_chunks'] = len([p for p in data_out.iterdir() if p.is_dir() and p.name.startswith('chunk-')])
    info['chunks_size'] = int(chunk_size)
    # set splits.train to 0:total_episodes (actual parquet count)
    splits = info.get('splits', {})
    splits['train'] = f"0:{len(new_episodes)}"
    info['splits'] = splits

    with open(meta_out / 'info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print('Wrote meta/info.json')

    # write merge report
    report = {
        'total_episodes_merged': len(new_episodes),
        'total_frames_merged': current_global_index,
        'total_tasks': total_tasks,
        'missing_parquet_count': missing_parquet_count,
        'unmatched_stats_count': unmatched_stats_count,
        'failures_sample': failures[:20],
        'unmatched_stats_examples': unmatched_stats_examples
    }
    try:
        with open(meta_out / 'merge_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print('Wrote meta/merge_report.json')
    except Exception as e:
        print('WARN: failed to write merge_report.json', e)

    if failures:
        print('Completed with failures:', len(failures))
        for it in failures[:20]:
            print(it)
        sys.exit(3)

    print('Merge complete: episodes=', len(new_episodes), 'frames=', current_global_index, 'tasks=', total_tasks)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input-repos', nargs='+', required=True, help='List of input repo paths (in desired concatenation order)')
    p.add_argument('--output-repo', required=True, help='Output merged repo path')
    p.add_argument('--chunk_size', default=1000, type=int, help='Number of episodes per chunk directory')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--auto_continue', action='store_true', help='On parquet read/write errors, continue instead of aborting')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    merge_repos(args.input_repos, args.output_repo, chunk_size=args.chunk_size, dry_run=args.dry_run, auto_continue=args.auto_continue)
