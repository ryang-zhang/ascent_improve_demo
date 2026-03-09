"""
Filter MP3D ObjectNav episodes into subsets for targeted evaluation.

Creates two filtered dataset splits:
  - val_cross_floor: 50 episodes with largest start-to-goal height difference (cross-floor)
  - val_long_distance: 50 episodes with largest geodesic distance

Usage:
    python scripts/filter_episodes.py [--num_episodes 50] [--height_threshold 1.5]

Output:
    data/datasets/objectnav/mp3d/v1/val_cross_floor/
    data/datasets/objectnav/mp3d/v1/val_long_distance/
"""

import argparse
import gzip
import json
import os
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path


def load_all_episodes(content_dir: str):
    """Load all episodes from per-scene content files, preserving source metadata."""
    all_episodes = []
    scene_metadata = {}

    for fname in sorted(os.listdir(content_dir)):
        if not fname.endswith(".json.gz"):
            continue
        filepath = os.path.join(content_dir, fname)
        with gzip.open(filepath, "rt") as f:
            data = json.load(f)

        scene_name = fname.replace(".json.gz", "")
        scene_metadata[scene_name] = {
            k: v for k, v in data.items() if k != "episodes"
        }

        for idx, ep in enumerate(data["episodes"]):
            ep["_source_scene"] = scene_name
            ep["_source_index"] = idx
            ep["_height_diff"] = abs(
                ep["start_position"][1] - ep["info"]["best_viewpoint_position"][1]
            )
            ep["_geo_dist"] = ep["info"].get("geodesic_distance", 0)
            all_episodes.append(ep)

    return all_episodes, scene_metadata


def select_cross_floor_episodes(episodes, num=50, height_threshold=1.5):
    """Select episodes with largest height difference between start and goal."""
    candidates = [ep for ep in episodes if ep["_height_diff"] > height_threshold]
    candidates.sort(key=lambda x: -x["_height_diff"])
    return candidates[:num]


def select_long_distance_episodes(episodes, num=50):
    """Select episodes with largest geodesic distance."""
    sorted_eps = sorted(episodes, key=lambda x: -x["_geo_dist"])
    return sorted_eps[:num]


def clean_episode(ep):
    """Remove internal metadata fields before saving."""
    cleaned = {k: v for k, v in ep.items() if not k.startswith("_")}
    return cleaned


def write_filtered_dataset(
    episodes, scene_metadata, base_dir, split_name, top_level_data
):
    """Write a filtered dataset in the same format as the original."""
    split_dir = os.path.join(base_dir, split_name)
    content_dir = os.path.join(split_dir, "content")
    os.makedirs(content_dir, exist_ok=True)

    by_scene = defaultdict(list)
    for ep in episodes:
        by_scene[ep["_source_scene"]].append(ep)

    for scene_name, scene_eps in by_scene.items():
        scene_data = deepcopy(scene_metadata.get(scene_name, {}))
        scene_data["episodes"] = [clean_episode(ep) for ep in scene_eps]

        out_path = os.path.join(content_dir, f"{scene_name}.json.gz")
        with gzip.open(out_path, "wt") as f:
            json.dump(scene_data, f)

    top_data = deepcopy(top_level_data)
    top_data["episodes"] = []
    top_path = os.path.join(split_dir, f"{split_name}.json.gz")
    with gzip.open(top_path, "wt") as f:
        json.dump(top_data, f)

    return split_dir


def print_summary(name, episodes):
    """Print summary statistics for a filtered set."""
    print(f"\n{'='*60}")
    print(f"  {name}: {len(episodes)} episodes")
    print(f"{'='*60}")

    scenes = set(ep["_source_scene"] for ep in episodes)
    categories = defaultdict(int)
    for ep in episodes:
        categories[ep["object_category"]] += 1

    geo_dists = [ep["_geo_dist"] for ep in episodes]
    height_diffs = [ep["_height_diff"] for ep in episodes]

    print(f"  Scenes: {len(scenes)} ({', '.join(sorted(scenes))})")
    print(f"  Geodesic dist: min={min(geo_dists):.2f}m, max={max(geo_dists):.2f}m, "
          f"mean={sum(geo_dists)/len(geo_dists):.2f}m")
    print(f"  Height diff:   min={min(height_diffs):.3f}m, max={max(height_diffs):.3f}m, "
          f"mean={sum(height_diffs)/len(height_diffs):.3f}m")
    print(f"  Categories: {dict(sorted(categories.items(), key=lambda x: -x[1]))}")

    print(f"\n  {'EP_ID':>6s}  {'SCENE':>15s}  {'HEIGHT_DIFF':>11s}  {'GEO_DIST':>10s}  CATEGORY")
    print(f"  {'-'*6}  {'-'*15}  {'-'*11}  {'-'*10}  {'-'*15}")
    for ep in episodes:
        print(f"  {ep['episode_id']:>6s}  {ep['_source_scene']:>15s}  "
              f"{ep['_height_diff']:>10.3f}m  {ep['_geo_dist']:>9.2f}m  {ep['object_category']}")


def main():
    parser = argparse.ArgumentParser(description="Filter MP3D ObjectNav episodes")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of episodes to select per category")
    parser.add_argument("--height_threshold", type=float, default=1.5,
                        help="Minimum height difference (m) for cross-floor episodes")
    parser.add_argument("--data_dir", type=str,
                        default="data/datasets/objectnav/mp3d/v1",
                        help="Base directory for ObjectNav MP3D dataset")
    parser.add_argument("--source_split", type=str, default="val",
                        help="Source split to filter from")
    args = parser.parse_args()

    source_dir = os.path.join(args.data_dir, args.source_split)
    content_dir = os.path.join(source_dir, "content")

    if not os.path.isdir(content_dir):
        print(f"Error: Content directory not found: {content_dir}")
        return

    print(f"Loading episodes from {content_dir}...")
    all_episodes, scene_metadata = load_all_episodes(content_dir)
    print(f"Loaded {len(all_episodes)} episodes from {len(scene_metadata)} scenes")

    top_level_path = os.path.join(source_dir, f"{args.source_split}.json.gz")
    with gzip.open(top_level_path, "rt") as f:
        top_level_data = json.load(f)

    cross_floor_eps = select_cross_floor_episodes(
        all_episodes, num=args.num_episodes, height_threshold=args.height_threshold
    )
    long_dist_eps = select_long_distance_episodes(
        all_episodes, num=args.num_episodes
    )

    print_summary("Cross-Floor Episodes", cross_floor_eps)
    print_summary("Long-Distance Episodes", long_dist_eps)

    cf_ids = set(id(ep) for ep in cross_floor_eps)
    ld_ids = set(id(ep) for ep in long_dist_eps)
    overlap = len(cf_ids & ld_ids)
    print(f"\nOverlap between two sets: {overlap} episodes")

    cf_dir = write_filtered_dataset(
        cross_floor_eps, scene_metadata, args.data_dir, "val_cross_floor", top_level_data
    )
    ld_dir = write_filtered_dataset(
        long_dist_eps, scene_metadata, args.data_dir, "val_long_distance", top_level_data
    )

    print(f"\nDatasets written:")
    print(f"  Cross-floor:   {cf_dir}")
    print(f"  Long-distance: {ld_dir}")
    print(f"\nTo evaluate on these subsets, use:")
    print(f"  habitat_baselines.eval.split=val_cross_floor")
    print(f"  habitat_baselines.eval.split=val_long_distance")


if __name__ == "__main__":
    main()
