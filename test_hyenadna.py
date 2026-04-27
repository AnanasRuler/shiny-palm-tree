"""Test HyenaDNA large-1m integration with SFFuseTask."""
import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from omegaconf import OmegaConf
from src.tasks.unified_task import SFFuseTask


def test_model_loading():
    """Test that HyenaDNA model loads correctly."""
    print("=" * 60)
    print("TEST 1: Model Loading")
    print("=" * 60)

    cfg = OmegaConf.load("configs/task/dualrep_hyenadna.yaml")
    task_cfg = cfg.task

    print(f"  pretrained_ckpt_path: {task_cfg.pretrained_ckpt_path}")
    print(f"  trainer.devices: {cfg.trainer.devices}")
    print(f"  trainer.accumulate_grad_batches: {cfg.trainer.accumulate_grad_batches}")
    print(f"  dataset.batch_size: {task_cfg.dataset.batch_size}")
    print(f"  dataset.num_workers: {task_cfg.dataset.num_workers}")

    task = SFFuseTask(
        encoder_config=task_cfg.encoder_config,
        bridge_config=task_cfg.bridge_config,
        decoder_config=task_cfg.decoder_config,
        head_config=task_cfg.head_config,
        n_pre_layers=task_cfg.n_pre_layers,
        pretrained_ckpt_path=task_cfg.pretrained_ckpt_path,
        freeze_encoder=False,
        use_lora=task_cfg.get("use_lora", False),
        lora_r=task_cfg.get("lora_r", 8),
        lora_alpha=task_cfg.get("lora_alpha", 16),
        lora_dropout=task_cfg.get("lora_dropout", 0.05),
        use_mlm_loss=False,
        dataset=task_cfg.dataset,
        optimizer_config=task_cfg.optimizer_config,
        scheduler_config=task_cfg.scheduler_config,
    )

    print(f"  Model type: {task._detect_model_type(task_cfg.pretrained_ckpt_path)}")
    print(f"  d_model: {task._d_model}")

    backbone = task._get_backbone()
    print(f"  Backbone: {type(backbone).__name__}, {len(backbone.layers)} layers")

    first_block = backbone.layers[0]
    print(f"  Block: {type(first_block).__name__}, fused_add_norm={hasattr(first_block, 'fused_add_norm')}")

    print("\n  Model loading: PASSED")
    return task


def test_single_gpu_forward(task):
    """Test single-sample forward pass (batch_size=1, as configured for each GPU)."""
    print("\n" + "=" * 60)
    print("TEST 2: Single GPU Forward (batch_size=1)")
    print("=" * 60)

    # Use CPU with reduced seq_len since GPU is incompatible (sm_120)
    device = torch.device("cpu")
    print("  Using CPU (GPU sm_120 incompatible with current PyTorch)")
    print("  Using reduced seq_len=16384 for CPU test")

    task = task.to(device)
    task.eval()

    seq_len = 16384  # 16384/128=128, still < 896 so head crop will fail. Use encoder forward only.
    input_ids = torch.randint(7, 12, (1, seq_len), dtype=torch.long, device=device)

    with torch.no_grad():
        print("  Running _encoder_forward...")
        enc_out = task._encoder_forward(input_ids)
        print(f"  Encoder output: {enc_out.shape}")
        assert enc_out.shape == (1, seq_len, task._d_model)

    print("\n  Single GPU forward: PASSED")


def test_block_signature(task):
    """Verify _run_backbone_layers routes to HyenaDNA path."""
    print("\n" + "=" * 60)
    print("TEST 3: Block Signature Detection")
    print("=" * 60)

    backbone = task._get_backbone()
    first_block = backbone.layers[0]
    has_fused = hasattr(first_block, 'fused_add_norm') and first_block.fused_add_norm

    device = torch.device("cpu")
    test_input = torch.randn(1, 128, task._d_model, device=device)

    with torch.no_grad():
        out_h, out_r = task._run_backbone_layers([first_block], test_input, None)

    assert out_r is None, "HyenaDNA should have residual=None"
    print(f"  fused_add_norm={has_fused}, residual_after=None, output={out_h.shape}")
    print("\n  Block signature: PASSED")


if __name__ == "__main__":
    torch.manual_seed(42)

    task = test_model_loading()
    test_single_gpu_forward(task)
    test_block_signature(task)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nNote: Full forward + backward test requires GPU with proper CUDA support.")
    print("Current GPU (sm_120) is incompatible with PyTorch cu130.")
    print("On a proper GPU, the config will use: 4 GPUs x batch_size=2 x accumulate=2 = effective_batch=16")
