"""Utilities for dealing with collection objects (lists, dicts) and configs."""

import functools
import json
import logging
import warnings
from typing import Sequence, Mapping, Callable

import hydra
import rich.syntax
import rich.tree
from omegaconf import ListConfig, DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


def is_list(x):
    """Check if x is a list-like object (but not a string)."""
    return isinstance(x, Sequence) and not isinstance(x, str)


def is_dict(x):
    """Check if x is a dict-like object."""
    return isinstance(x, Mapping)


def to_dict(x, recursive=True):
    """Convert Sequence or Mapping object to dict.
    
    Lists get converted to {0: x[0], 1: x[1], ...}
    """
    if is_list(x):
        x = {i: v for i, v in enumerate(x)}
    if is_dict(x):
        if recursive:
            return {k: to_dict(v, recursive=recursive) for k, v in x.items()}
        else:
            return dict(x)
    else:
        return x


def to_list(x, recursive=False):
    """Convert an object to list.

    If Sequence (e.g. list, tuple, Listconfig): just return it
    Special case: If non-recursive and not a list, wrap in list
    """
    if is_list(x):
        if recursive:
            return [to_list(_x) for _x in x]
        else:
            return list(x)
    else:
        if recursive:
            return x
        else:
            return [x]


def omegaconf_filter_keys(d, fn=None):
    """Only keep keys where fn(key) is True. Support nested DictConfig."""
    if fn is None:
        def fn(x):
            return True
    if is_list(d):
        return ListConfig([omegaconf_filter_keys(v, fn) for v in d])
    elif is_dict(d):
        return DictConfig(
            {k: omegaconf_filter_keys(v, fn) for k, v in d.items() if fn(k)}
        )
    else:
        return d


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # This ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level_name in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level_name, rank_zero_only(getattr(logger, level_name)))

    return logger


def process_config(config: DictConfig) -> DictConfig:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    
    Args:
        config: Configuration composed by Hydra.
    
    Returns:
        Processed config.
    """
    log = get_logger()

    # Filter out keys that were used just for interpolation
    config = omegaconf_filter_keys(config, lambda k: not k.startswith('__'))

    # Enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # Disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

        # Force debugger friendly configuration
        log.info("Forcing debugger friendly configuration!")
        if config.trainer.get("accelerator") == "gpu":
            config.trainer.accelerator = "cpu"
            config.trainer.devices = 1
        if config.get("loader", {}).get("pin_memory"):
            config.loader.pin_memory = False
        if config.get("loader", {}).get("num_workers"):
            config.loader.num_workers = 0

    return config


@rank_zero_only
def print_config(
    config: DictConfig,
    resolve: bool = True,
    save_cfg: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    
    Args:
        config: Configuration composed by Hydra.
        resolve: Whether to resolve reference fields of DictConfig.
        save_cfg: Whether to save the config to a file.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    if save_cfg:
        with open("config_tree.txt", "w") as fp:
            rich.print(tree, file=fp)
        with open("config.json", "w") as fp:
            json.dump(OmegaConf.to_container(config, resolve=True), fp, indent=4)


def print_model_info(model, config=None):
    """Print detailed model information."""
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE AND CONFIGURATION")
    print("=" * 80)
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # Print component parameters
    if hasattr(model, 'encoder'):
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        print(f"\nEncoder Parameters: {encoder_params:,} (trainable: {encoder_trainable:,})")
    
    if hasattr(model, 'head'):
        head_params = sum(p.numel() for p in model.head.parameters())
        print(f"Head Parameters: {head_params:,}")
    
    if hasattr(model, 'mlm_projection'):
        mlm_params = sum(p.numel() for p in model.mlm_projection.parameters())
        print(f"MLM Projection Parameters: {mlm_params:,}")
    
    print("\n" + "-" * 80)
    print("MODULE BREAKDOWN:")
    print("-" * 80)
    
    # Print top-level modules
    for name, module in model.named_children():
        if hasattr(module, 'parameters'):
            module_params = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {module_params:,} parameters")
    
    print("=" * 80 + "\n")


def instantiate(registry, config, *args, partial=False, wrap=None, **kwargs):
    """Instantiate an object from registry.
    
    Args:
        registry: Dictionary mapping names to functions or target paths.
        config: Dictionary with a '_name_' key indicating which element of the registry to grab.
        *args, **kwargs: Additional arguments to pass into the target constructor.
        partial: If True, return partial function instead of instantiated object.
        wrap: Wrap the target class.
    
    Returns:
        Instantiated object or partial function.
    """
    # Case 1: no config
    if config is None:
        return None
    
    # Case 2a: string means _name_ was overloaded
    if isinstance(config, str):
        _name_ = None
        _target_ = registry[config]
        config = {}
    # Case 2b: grab the desired callable from name
    else:
        _name_ = config.pop("_name_")
        _target_ = registry[_name_]

    # Retrieve the right constructor automatically based on type
    if isinstance(_target_, str):
        fn = hydra.utils.get_method(path=_target_)
    elif isinstance(_target_, Callable):
        fn = _target_
    else:
        raise NotImplementedError("instantiate target must be string or callable")

    # Instantiate object
    if wrap is not None:
        fn = wrap(fn)
    obj = functools.partial(fn, *args, **config, **kwargs)

    # Restore _name_
    if _name_ is not None:
        config["_name_"] = _name_

    if partial:
        return obj
    else:
        return obj()
