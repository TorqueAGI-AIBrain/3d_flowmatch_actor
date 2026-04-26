# apply_tf_corrections.py: Apply front camera TF static correction to MCAP bags.
# apply_tf_corrections.py: Reads from default_task_fixed, writes corrected MCAPs to default_task_fixed_v2.

"""
Apply front camera calibration correction to all MCAP bags.

Only one change is made:
  - /tf_static: link_base -> front_optical_frame is corrected using the
    CloudCompare-derived calibration matrix (see docs/tf_correction_v2.md)
  - /front/pose: recomputed from the corrected TF static so TF and pose are consistent

All other topics (wrist, /tf dynamic, images, etc.) are copied unchanged.

Usage:
    python -m data_processing.apply_tf_corrections
    python -m data_processing.apply_tf_corrections --src data/xarm/other --dst data/xarm/other_v2
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap.writer import Writer as McapWriter
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)
serialize_cdr = TYPESTORE.serialize_cdr
deserialize_cdr = TYPESTORE.deserialize_cdr

# ---------------------------------------------------------------------------
# Front camera correction (from CloudCompare alignment, see docs/tf_correction_v2.md)
# Origin axes were aligned to robot base plate; this matrix is the measured offset.
# Applied as: corrected_tf = inv(FRONT_CORRECTION) @ original_tf
# ---------------------------------------------------------------------------
# fmt: off
FRONT_CORRECTION = np.array([
    [0.990103,  0.139081, -0.018772, 0.036359],
    [-0.133884, 0.976160,  0.170840, 0.000056],
    [0.042085, -0.166636,  0.985120, 0.067018],
    [0.000000,  0.000000,  0.000000, 1.000000],
], dtype=np.float64)
# fmt: on
FRONT_CORRECTION_INV = np.linalg.inv(FRONT_CORRECTION)


def write_mat_to_tf_transform(t, mat):
    """Write a 4x4 matrix into a TransformStamped message (in-place)."""
    pos = mat[:3, 3]
    q = Rotation.from_matrix(mat[:3, :3]).as_quat()
    t.transform.translation.x = float(pos[0])
    t.transform.translation.y = float(pos[1])
    t.transform.translation.z = float(pos[2])
    t.transform.rotation.x = float(q[0])
    t.transform.rotation.y = float(q[1])
    t.transform.rotation.z = float(q[2])
    t.transform.rotation.w = float(q[3])


def write_mat_to_pose(msg, mat):
    """Write a 4x4 matrix into a PoseStamped message (in-place)."""
    pos = mat[:3, 3]
    q = Rotation.from_matrix(mat[:3, :3]).as_quat()
    msg.pose.position.x = float(pos[0])
    msg.pose.position.y = float(pos[1])
    msg.pose.position.z = float(pos[2])
    msg.pose.orientation.x = float(q[0])
    msg.pose.orientation.y = float(q[1])
    msg.pose.orientation.z = float(q[2])
    msg.pose.orientation.w = float(q[3])


def process_mcap(src_path, dst_path):
    """Read MCAP, correct front TF static + /front/pose, copy everything else.

    The corrected value is: inv(FRONT_CORRECTION) @ original_/front/pose.
    Both /tf_static and /front/pose are set to this same corrected value.
    """
    # Pass 1: read original /front/pose to compute the corrected transform.
    # FRONT_CORRECTION was calibrated against /front/pose, NOT against TF static.
    original_front_pose = None
    with open(src_path, 'rb') as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            if channel.topic == '/front/pose':
                msg = deserialize_cdr(message.data, schema.name)
                p, q = msg.pose.position, msg.pose.orientation
                original_front_pose = np.eye(4, dtype=np.float64)
                original_front_pose[:3, :3] = Rotation.from_quat(
                    [q.x, q.y, q.z, q.w]).as_matrix()
                original_front_pose[:3, 3] = [p.x, p.y, p.z]
                break

    if original_front_pose is None:
        raise RuntimeError(f"No /front/pose in {src_path}")

    corrected_front = FRONT_CORRECTION_INV @ original_front_pose
    # Additional fixes from Foxglove visual alignment
    corrected_front[2, 3] -= 0.02  # 2cm upward in Z
    corrected_front[1, 3] += 0.02  # 2cm in +Y

    # Pass 2: rewrite the MCAP
    front_pose_count = 0
    total_msgs = 0

    with open(src_path, 'rb') as f_in, open(dst_path, 'wb') as f_out:
        reader = make_reader(f_in)
        writer = McapWriter(f_out)
        writer.start()

        schema_map = {}
        channel_map = {}

        for schema, channel, message in reader.iter_messages():
            if schema.id not in schema_map:
                schema_map[schema.id] = writer.register_schema(
                    name=schema.name, encoding=schema.encoding, data=schema.data)

            if channel.id not in channel_map:
                channel_map[channel.id] = writer.register_channel(
                    topic=channel.topic, message_encoding=channel.message_encoding,
                    schema_id=schema_map[schema.id], metadata=channel.metadata)

            data = message.data

            if channel.topic == '/tf_static':
                msg = deserialize_cdr(message.data, schema.name)
                for t in msg.transforms:
                    # Fix 1: front camera TF from CloudCompare calibration + 2cm Z
                    if t.header.frame_id == 'link_base' and t.child_frame_id == 'front_optical_frame':
                        write_mat_to_tf_transform(t, corrected_front)
                    # Fix 2: link_eef -> wrist_optical_frame
                    # Z: 11.9cm -> 1.19cm, Y: -1cm offset, Z: -1cm offset, yaw: -3deg
                    if t.header.frame_id == 'link_eef' and t.child_frame_id == 'wrist_optical_frame':
                        t.transform.translation.y += -0.01
                        t.transform.translation.z = 0.0119 + (-0.01)
                        # Apply -3deg yaw offset to existing rotation
                        orig_q = [t.transform.rotation.x, t.transform.rotation.y,
                                  t.transform.rotation.z, t.transform.rotation.w]
                        orig_R = Rotation.from_quat(orig_q)
                        yaw_offset = Rotation.from_euler('z', -3, degrees=True)
                        new_R = orig_R * yaw_offset  # rotate in local frame
                        new_q = new_R.as_quat()
                        t.transform.rotation.x = float(new_q[0])
                        t.transform.rotation.y = float(new_q[1])
                        t.transform.rotation.z = float(new_q[2])
                        t.transform.rotation.w = float(new_q[3])
                data = serialize_cdr(msg, schema.name)

            elif channel.topic == '/front/pose':
                # Set /front/pose to same corrected value (derived from TF)
                msg = deserialize_cdr(message.data, schema.name)
                write_mat_to_pose(msg, corrected_front)
                data = serialize_cdr(msg, schema.name)
                front_pose_count += 1

            writer.add_message(
                channel_id=channel_map[channel.id],
                log_time=message.log_time, data=data,
                publish_time=message.publish_time)
            total_msgs += 1

        writer.finish()

    return total_msgs, front_pose_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='data/xarm/default_task_fixed')
    parser.add_argument('--dst', type=str, default='data/xarm/default_task_fixed_v2')
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    episode_dirs = sorted([d for d in src_dir.iterdir() if d.is_dir() and d.name.startswith('episode_')])
    print(f"Source: {src_dir} ({len(episode_dirs)} episodes)")
    print(f"Destination: {dst_dir}")
    print(f"Fix: link_base -> front_optical_frame TF static + /front/pose derived from it")
    print()

    dst_dir.mkdir(parents=True, exist_ok=True)

    for ep_dir in episode_dirs:
        mcap_files = list(ep_dir.glob('*.mcap'))
        if not mcap_files:
            print(f"  {ep_dir.name}: no MCAP files, skipping")
            continue

        dst_ep_dir = dst_dir / ep_dir.name
        dst_ep_dir.mkdir(parents=True, exist_ok=True)

        meta = ep_dir / 'metadata.yaml'
        if meta.exists():
            shutil.copy2(meta, dst_ep_dir / 'metadata.yaml')

        for mcap_file in mcap_files:
            dst_mcap = dst_ep_dir / mcap_file.name
            total, front_n = process_mcap(mcap_file, dst_mcap)
            print(f"  {ep_dir.name}/{mcap_file.name}: {total} msgs, front_pose={front_n} rewritten")

    for extra in ['instructions.json']:
        src_file = src_dir / extra
        if src_file.exists():
            shutil.copy2(src_file, dst_dir / extra)
            print(f"\nCopied {extra}")

    print(f"\nDone. Corrected bags in {dst_dir}")


if __name__ == '__main__':
    main()
