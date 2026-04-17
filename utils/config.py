# config.py: YAML configuration loader with CLI override support.
# config.py: Loads a YAML config and flattens it into argparse-compatible args.

import argparse
import yaml
from pathlib import Path


def load_yaml_config(config_path):
    """Load a YAML config file and return as nested dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def flatten_config(config, parent_key='', sep='_'):
    """Flatten nested dict into a single-level dict with joined keys."""
    items = {}
    for k, v in config.items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_config(v, key, sep))
        else:
            items[key] = v
    return items


def _apply_type(value, type_fn):
    """Convert a value using the given type function, handling None and bool."""
    if value is None:
        return None
    if type_fn is bool or type_fn.__name__ == 'str2bool':
        if isinstance(value, bool):
            return value
        return str(value).lower() in ('true', '1', 'yes', 't', 'y')
    return type_fn(value)


def config_to_args(config, arg_spec):
    """
    Map a flat config dict to argparse-style namespace values.

    Args:
        config: flat dict from flatten_config()
        arg_spec: list of (name, type, default) tuples from main.py's argument list

    Returns:
        dict of {arg_name: typed_value}
    """
    # Build lookup from arg_spec
    spec_map = {name: (type_fn, default) for name, type_fn, default in arg_spec}
    result = {}

    for name, (type_fn, default) in spec_map.items():
        if name in config:
            result[name] = _apply_type(config[name], type_fn)
        # else: leave unset, argparse default will handle it

    return result


def add_config_arg(parser):
    """Add --config argument to an existing argparse parser."""
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML config file. CLI args override config values.'
    )
    return parser


def merge_config_and_cli(config_path, parser, cli_args=None):
    """
    Load YAML config, parse CLI args, and merge with CLI taking precedence.

    Args:
        config_path: path to YAML config file (or None)
        parser: argparse.ArgumentParser with all arguments defined
        cli_args: optional list of CLI args (defaults to sys.argv)

    Returns:
        argparse.Namespace with merged values
    """
    if config_path is None:
        return parser.parse_args(cli_args)

    config = load_yaml_config(config_path)
    flat = flatten_config(config)

    # Parse CLI args to find explicit overrides
    args, _ = parser.parse_known_args(cli_args)

    # Set config values as defaults, then re-parse so CLI wins
    defaults = {}
    for key, value in flat.items():
        if hasattr(args, key):
            defaults[key] = value

    parser.set_defaults(**defaults)
    return parser.parse_args(cli_args)


def load_config(config_path):
    """
    Simple config loader for non-argparse scripts (zarr converter, bag extractor).

    Args:
        config_path: path to YAML config file

    Returns:
        argparse.Namespace with config values as attributes
    """
    config = load_yaml_config(config_path)
    flat = flatten_config(config)
    return argparse.Namespace(**flat)


def load_config_nested(config_path):
    """
    Load config and return as nested namespace for scripts that prefer
    grouped access (e.g., config.cameras.front.rgb_topic).

    Args:
        config_path: path to YAML config file

    Returns:
        nested SimpleNamespace
    """
    from types import SimpleNamespace

    def _to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
        if isinstance(d, list):
            return [_to_namespace(item) for item in d]
        return d

    config = load_yaml_config(config_path)
    return _to_namespace(config)
