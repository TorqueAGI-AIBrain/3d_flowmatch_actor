"""
mix_zarr.py: Mix bookshelf (Hiveformer) and toolbox (xArm real) zarr datasets.

Randomly samples a subset of bookshelf episodes and combines with all toolbox
episodes into a single mixed zarr store. Provenance is tracked in split_info.json.

Usage:
    python -m data_processing.mix_zarr --config configs/mix_zarr.yaml
"""

import argparse
import json
import os

import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Mix two zarr datasets (bookshelf + toolbox) for multi-task training"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    return parser.parse_args()


def _resolve_config(config_path):
    from utils.config import load_yaml_config
    return load_yaml_config(config_path)


def _open_zarr(path):
    return zarr.open_group(path, mode="r")


def _create_zarr(path, src_zarr):
    """Create a new zarr store matching the schema of src_zarr."""
    compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)
    out = zarr.open_group(path, mode="w")
    for key in src_zarr.keys():
        shape = src_zarr[key].shape[1:]  # drop leading N
        out.create_dataset(
            key,
            shape=(0,) + shape,
            chunks=(1,) + shape,
            compressor=compressor,
            dtype=src_zarr[key].dtype,
        )
    return out


def _append_sample(out_zarr, src_zarr, idx, task_id_override=None):
    """Append one sample (row idx) from src_zarr to out_zarr."""
    for key in src_zarr.keys():
        val = src_zarr[key][idx:idx + 1]
        if key == "task_id" and task_id_override is not None:
            val = np.array([task_id_override], dtype=np.uint8)
        out_zarr[key].append(val)


def _sample_bookshelf_indices(n_total, n_samples, seed):
    """Randomly select n_samples indices from [0, n_total)."""
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(n_total, size=n_samples, replace=False).tolist())


def main():
    args = parse_arguments()
    cfg = _resolve_config(args.config)

    bookshelf_train = cfg["bookshelf_train_zarr"]
    bookshelf_val = cfg["bookshelf_val_zarr"]
    toolbox_train = cfg["toolbox_train_zarr"]
    toolbox_val = cfg["toolbox_val_zarr"]
    output_dir = cfg["output_dir"]
    n_bookshelf_train = cfg["n_bookshelf_train_samples"]
    n_bookshelf_val = cfg.get("n_bookshelf_val_samples", 10)
    seed = cfg.get("seed", 42)
    bookshelf_task_name = cfg["bookshelf_task_name"]
    toolbox_task_name = cfg["toolbox_task_name"]

    os.makedirs(output_dir, exist_ok=True)

    # Task ID assignment: bookshelf=0, toolbox=1
    task2id = {bookshelf_task_name: 0, toolbox_task_name: 1}
    print(f"Task IDs: {task2id}")

    for split in ["train", "val"]:
        out_path = os.path.join(output_dir, f"{split}.zarr")
        if os.path.exists(out_path):
            print(f"{out_path} already exists, skipping. Delete to regenerate.")
            continue

        if split == "train":
            bk_zarr = _open_zarr(bookshelf_train)
            tb_zarr = _open_zarr(toolbox_train)
            n_bk = n_bookshelf_train
        else:
            bk_zarr = _open_zarr(bookshelf_val)
            tb_zarr = _open_zarr(toolbox_val)
            n_bk = n_bookshelf_val

        n_bk_total = bk_zarr["action"].shape[0]
        n_tb_total = tb_zarr["action"].shape[0]

        # Randomly sample bookshelf indices
        bk_indices = _sample_bookshelf_indices(n_bk_total, min(n_bk, n_bk_total), seed)

        # All toolbox indices
        tb_indices = list(range(n_tb_total))

        print(f"\n{split}: {len(bk_indices)} bookshelf + {len(tb_indices)} toolbox samples")

        # Verify schema compatibility (action shape must match)
        bk_action_shape = bk_zarr["action"].shape[1:]
        tb_action_shape = tb_zarr["action"].shape[1:]
        assert bk_action_shape == tb_action_shape, (
            f"Action shape mismatch: bookshelf={bk_action_shape}, toolbox={tb_action_shape}. "
            f"Re-extract toolbox zarr with matching trajectory_length."
        )

        # Create output zarr matching the bookshelf schema
        out_zarr = _create_zarr(out_path, bk_zarr)

        # Append bookshelf samples
        for idx in tqdm(bk_indices, desc=f"  bookshelf ({split})"):
            _append_sample(out_zarr, bk_zarr, idx, task_id_override=task2id[bookshelf_task_name])

        # Append toolbox samples
        for idx in tqdm(tb_indices, desc=f"  toolbox ({split})"):
            _append_sample(out_zarr, tb_zarr, idx, task_id_override=task2id[toolbox_task_name])

        print(f"  Written {out_zarr['action'].shape[0]} samples to {out_path}")

        # Save provenance split info per split
        split_info = {
            "seed": seed,
            "task_ids": task2id,
            "bookshelf": {
                "source": bookshelf_train if split == "train" else bookshelf_val,
                "total_available": n_bk_total,
                "sampled_indices": bk_indices,
                "n_sampled": len(bk_indices),
            },
            "toolbox": {
                "source": toolbox_train if split == "train" else toolbox_val,
                "total_available": n_tb_total,
                "sampled_indices": tb_indices,
                "n_sampled": len(tb_indices),
            },
        }
        split_info_path = os.path.join(output_dir, f"split_info_{split}.json")
        with open(split_info_path, "w") as f:
            json.dump(split_info, f, indent=2)
        print(f"  Provenance saved to {split_info_path}")

    # Build merged instructions.json
    # Bookshelf: single variation "0" with task descriptions
    # Toolbox: variation "0" with "place object in toolbox"
    instructions = {
        bookshelf_task_name: {
            "0": cfg.get("bookshelf_instructions", ["put 1 books on bookshelf",
                                                     "pick up 1 books and place them on the top shelf",
                                                     "stack 1 books up on the top shelf"])
        },
        toolbox_task_name: {
            "0": cfg.get("toolbox_instructions", ["place object in toolbox"])
        },
    }
    instr_dir = os.path.join("instructions", "xarm")
    os.makedirs(instr_dir, exist_ok=True)
    instr_path = os.path.join(instr_dir, "instructions_mixed.json")
    with open(instr_path, "w") as f:
        json.dump(instructions, f, indent=2)
    print(f"\nMerged instructions saved to {instr_path}")

    print("\n=== Ready for training ===")
    print(f"  train_data_dir: {os.path.join(output_dir, 'train.zarr')}")
    print(f"  eval_data_dir:  {os.path.join(output_dir, 'val.zarr')}")
    print(f"  train_instructions: {instr_path}")
    print(f"  task_ids: {task2id}")


if __name__ == "__main__":
    main()
