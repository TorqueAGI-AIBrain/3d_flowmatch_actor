# xarm.py: Zarr-based dataset for xArm real robot demonstrations.
# xarm.py: Reads from zarr files produced by data_processing/xarm_to_zarr.py.

import json
import random

from .base import BaseDataset


class XArmDataset(BaseDataset):
    """Dataset for xArm demonstrations stored in zarr format."""
    quat_format = 'xyzw'
    camera_inds = None
    camera_inds2d = None
    train_copies = 10

    def __init__(
        self,
        root,
        instructions,
        copies=None,
        relative_action=False,
        mem_limit=8,
        actions_only=False,
        chunk_size=1
    ):
        with open(instructions) as f:
            instr = json.load(f)
        self.tasks = list(instr.keys())
        self.cameras = None

        super().__init__(
            root=root,
            instructions=instructions,
            copies=copies,
            relative_action=relative_action,
            mem_limit=mem_limit,
            actions_only=actions_only,
            chunk_size=chunk_size
        )

    def _get_task(self, idx):
        return [
            self.tasks[int(tid)]
            for tid in self.annos['task_id'][idx:idx + self.chunk_size]
        ]

    def _get_instr(self, idx):
        return [
            random.choice(self._instructions[self.tasks[int(t)]][str(int(v))])
            for t, v in zip(
                self.annos['task_id'][idx:idx + self.chunk_size],
                self.annos['variation'][idx:idx + self.chunk_size]
            )
        ]

    def _get_rgb2d(self, idx):
        return None

    def _get_extrinsics(self, idx):
        return self._get_attr_by_idx(idx, 'extrinsics', False)

    def _get_intrinsics(self, idx):
        return self._get_attr_by_idx(idx, 'intrinsics', False)

    def __getitem__(self, idx):
        idx = idx % (len(self.annos['action']) // self.chunk_size)
        idx = idx * self.chunk_size
        if self._actions_only:
            return {"action": self._get_action(idx)}
        return {
            "task": self._get_task(idx),
            "instr": self._get_instr(idx),
            "rgb": self._get_rgb(idx),
            "depth": self._get_depth(idx),
            "rgb2d": self._get_rgb2d(idx),
            "proprioception": self._get_proprioception(idx),
            "action": self._get_action(idx),
            "extrinsics": self._get_extrinsics(idx),
            "intrinsics": self._get_intrinsics(idx),
        }
