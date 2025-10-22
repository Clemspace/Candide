"""Checkpoint saving and loading utilities."""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    epoch: int = 0,
    loss: float = 0.0,
    config: Optional[Dict[str, Any]] = None,
    save_path: Union[str, Path] = "checkpoint.pt",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state (optional)
        scheduler: Learning rate scheduler (optional)
        step: Training step number
        epoch: Training epoch number
        loss: Current loss value
        config: Model configuration dict
        save_path: Path to save checkpoint
        metadata: Additional metadata to save
        
    Examples:
        >>> save_checkpoint(
        ...     model, optimizer, scheduler,
        ...     step=5000, loss=2.5,
        ...     save_path="checkpoints/step_5000.pt"
        ... )
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model': model.state_dict(),
        'step': step,
        'epoch': epoch,
        'loss': loss,
    }
    
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    
    if config is not None:
        checkpoint['config'] = config
    
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    # Save
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into (optional)
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to map tensors to
        strict: Whether to strictly enforce state dict keys match
        
    Returns:
        Dictionary containing checkpoint data
        
    Examples:
        >>> # Load checkpoint info only
        >>> checkpoint = load_checkpoint("checkpoint.pt")
        >>> print(f"Step: {checkpoint['step']}")
        
        >>> # Load into model
        >>> checkpoint = load_checkpoint(
        ...     "checkpoint.pt",
        ...     model=model,
        ...     optimizer=optimizer,
        ...     device=device
        ... )
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    if model is not None:
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume checkpoint is the state dict itself
            state_dict = checkpoint
            checkpoint = {'model': state_dict}
        
        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        
        if missing_keys:
            logger.warning(f"Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in state dict: {unexpected_keys}")
        
        logger.info("Model weights loaded")
    
    # Load optimizer state
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Optimizer state loaded")
    
    # Load scheduler state
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("Scheduler state loaded")
    
    # Log checkpoint info
    if 'step' in checkpoint:
        logger.info(f"Checkpoint step: {checkpoint['step']}")
    if 'epoch' in checkpoint:
        logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        logger.info(f"Checkpoint loss: {checkpoint['loss']:.4f}")
    
    return checkpoint


def load_model_only(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> torch.nn.Module:
    """Load only model weights from checkpoint (convenience function).
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        device: Device to map tensors to
        strict: Whether to strictly enforce state dict keys match
        
    Returns:
        Model with loaded weights
        
    Examples:
        >>> model = create_model(config)
        >>> model = load_model_only("checkpoint.pt", model, device)
    """
    load_checkpoint(checkpoint_path, model=model, device=device, strict=strict)
    return model


def get_checkpoint_info(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """Get information about a checkpoint without loading weights.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary with checkpoint metadata
        
    Examples:
        >>> info = get_checkpoint_info("checkpoint.pt")
        >>> print(f"Step: {info.get('step', 'unknown')}")
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    info = {
        'path': str(checkpoint_path),
        'size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
    }
    
    # Extract metadata
    for key in ['step', 'epoch', 'loss', 'config', 'metadata']:
        if key in checkpoint:
            info[key] = checkpoint[key]
    
    # Check what's in the checkpoint
    info['has_model'] = any(k in checkpoint for k in ['model', 'model_state_dict', 'state_dict'])
    info['has_optimizer'] = 'optimizer' in checkpoint
    info['has_scheduler'] = 'scheduler' in checkpoint
    
    return info


# Self-test
if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    
    print("Testing checkpoint utilities...")
    
    # Create a dummy model
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Test save
    print("\n1. Testing save_checkpoint...")
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_checkpoint.pt"
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=100,
            epoch=5,
            loss=2.5,
            config={'dim': 10},
            save_path=save_path,
        )
        
        print(f"   ✓ Checkpoint saved to {save_path}")
        print(f"   ✓ File size: {save_path.stat().st_size / 1024:.2f} KB")
        
        # Test load
        print("\n2. Testing load_checkpoint...")
        checkpoint = load_checkpoint(save_path)
        print(f"   ✓ Step: {checkpoint['step']}")
        print(f"   ✓ Epoch: {checkpoint['epoch']}")
        print(f"   ✓ Loss: {checkpoint['loss']}")
        
        # Test load into model
        print("\n3. Testing load into model...")
        new_model = torch.nn.Linear(10, 10)
        load_checkpoint(save_path, model=new_model, optimizer=optimizer)
        print(f"   ✓ Weights loaded into model")
        
        # Test checkpoint info
        print("\n4. Testing get_checkpoint_info...")
        info = get_checkpoint_info(save_path)
        print(f"   ✓ Has model: {info['has_model']}")
        print(f"   ✓ Has optimizer: {info['has_optimizer']}")
        print(f"   ✓ Size: {info['size_mb']:.2f} MB")
    
    print("\n✅ All checkpoint tests passed!")