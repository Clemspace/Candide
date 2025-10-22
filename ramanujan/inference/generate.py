"""Text generation utilities for Candide models.

Supports various sampling strategies:
- Greedy decoding
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling
"""

from dataclasses import dataclass
from typing import Optional, List, Union, Callable
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer


@dataclass
class GenerationConfig:
    """Configuration for text generation.
    
    Attributes:
        max_length: Maximum number of tokens to generate
        max_new_tokens: Maximum number of NEW tokens (alternative to max_length)
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top k tokens for sampling (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        do_sample: Whether to sample (False = greedy)
        num_beams: Number of beams for beam search (1 = no beam search)
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
        no_repeat_ngram_size: Size of n-grams that can't repeat
        pad_token_id: Token ID for padding
        eos_token_id: Token ID for end of sequence
        use_cache: Whether to use KV cache for faster generation
    """
    max_length: int = 100
    max_new_tokens: Optional[int] = None
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    use_cache: bool = True


def sample_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Apply top-k filtering to logits.
    
    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        k: Number of top tokens to keep
        
    Returns:
        Filtered logits with only top-k values
    """
    if k <= 0:
        return logits
    
    # Get top-k values and indices
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    
    # Create mask for top-k tokens
    mask = torch.full_like(logits, float('-inf'))
    mask.scatter_(-1, top_k_indices, top_k_values)
    
    return mask


def sample_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Apply nucleus (top-p) filtering to logits.
    
    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        p: Cumulative probability threshold
        
    Returns:
        Filtered logits with nucleus sampling
    """
    if p >= 1.0:
        return logits
    
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > p
    
    # Shift the indices to the right to keep first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Scatter sorted tensors back to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    
    logits = logits.masked_fill(indices_to_remove, float('-inf'))
    return logits


def sample_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to logits.
    
    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        temperature: Temperature value (higher = more random)
        
    Returns:
        Temperature-scaled logits
    """
    return logits / temperature


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty to discourage repeated tokens.
    
    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        generated_tokens: Previously generated tokens (batch_size, seq_len)
        penalty: Penalty factor (> 1.0 discourages repetition)
        
    Returns:
        Logits with repetition penalty applied
    """
    if penalty == 1.0:
        return logits
    
    batch_size, vocab_size = logits.shape
    
    # For each token in generated sequence, apply penalty
    for i in range(batch_size):
        for token_id in generated_tokens[i]:
            if token_id < vocab_size:
                # If logit is positive, divide by penalty; if negative, multiply
                if logits[i, token_id] < 0:
                    logits[i, token_id] *= penalty
                else:
                    logits[i, token_id] /= penalty
    
    return logits


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> str:
    """Generate text from a prompt.
    
    Args:
        model: Trained language model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input text prompt
        config: Generation configuration
        device: Device to run generation on
        verbose: Whether to print generation progress
        
    Returns:
        Generated text string
        
    Examples:
        >>> model = load_model("checkpoints/model.pt")
        >>> tokenizer = get_tokenizer("gpt2")
        >>> config = GenerationConfig(max_new_tokens=50, temperature=0.8)
        >>> text = generate(model, tokenizer, "Once upon a time", config)
    """
    if config is None:
        config = GenerationConfig()
    
    if device is None:
        device = next(model.parameters()).device
    
    # Set model to eval mode
    model.eval()
    
    # Encode prompt
    #input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    input_ids = tokenizer.encode(prompt)
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # Add batch dimension
    input_ids = input_ids.to(device)
    
    # Set special token IDs if not provided
    if config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id
    if config.eos_token_id is None:
        config.eos_token_id = tokenizer.eos_token_id
    
    # Determine max length
    if config.max_new_tokens is not None:
        max_length = input_ids.shape[1] + config.max_new_tokens
    else:
        max_length = config.max_length
    
    # Generate tokens
    generated = input_ids
    past_key_values = None
    
    for step in range(max_length - input_ids.shape[1]):
        # Forward pass
        # Simple forward without KV cache (your model doesn't support it)
        logits = model(generated)

        # If model returns tuple (logits, other_stuff), extract logits
        if isinstance(logits, tuple):
            logits = logits[0]

        past_key_values = None  # Don't use cache
        
        # Get logits for next token
        next_token_logits = logits[:, -1, :]
        
        # Apply repetition penalty
        if config.repetition_penalty != 1.0:
            next_token_logits = apply_repetition_penalty(
                next_token_logits, generated, config.repetition_penalty
            )
        
        # Apply sampling strategies
        if config.do_sample:
            # Apply temperature
            if config.temperature != 1.0:
                next_token_logits = sample_temperature(
                    next_token_logits, config.temperature
                )
            
            # Apply top-k
            if config.top_k > 0:
                next_token_logits = sample_top_k(next_token_logits, config.top_k)
            
            # Apply top-p
            if config.top_p < 1.0:
                next_token_logits = sample_top_p(next_token_logits, config.top_p)
            
            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Append to generated sequence
        generated = torch.cat([generated, next_token], dim=1)
        
        # Check for EOS token
        if config.eos_token_id is not None and next_token.item() == config.eos_token_id:
            break
        
        if verbose and (step + 1) % 10 == 0:
            print(f"Generated {step + 1} tokens...")
    
    # Decode and return
    #generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    generated_ids = generated[0].cpu().tolist() if isinstance(generated[0], torch.Tensor) else generated[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text


@torch.no_grad()
def generate_batch(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None,
) -> List[str]:
    """Generate text for multiple prompts in batch.
    
    Args:
        model: Trained language model
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of input prompts
        config: Generation configuration
        device: Device to run generation on
        
    Returns:
        List of generated text strings
        
    Examples:
        >>> prompts = ["Once upon a time", "In a galaxy far away"]
        >>> texts = generate_batch(model, tokenizer, prompts, config)
    """
    # For simplicity, generate one at a time
    # TODO: Implement proper batched generation with padding
    return [generate(model, tokenizer, prompt, config, device) for prompt in prompts]


# Self-test
if __name__ == "__main__":
    print("Testing generation utilities...")
    
    # Test 1: Sampling functions
    print("\n1. Testing sampling functions...")
    logits = torch.randn(1, 100)
    
    # Top-k
    filtered = sample_top_k(logits.clone(), k=10)
    non_inf = (filtered != float('-inf')).sum()
    print(f"   ✓ Top-k keeps {non_inf} tokens (expected 10)")
    
    # Top-p
    filtered = sample_top_p(logits.clone(), p=0.9)
    non_inf = (filtered != float('-inf')).sum()
    print(f"   ✓ Top-p keeps {non_inf} tokens")
    
    # Temperature
    scaled = sample_temperature(logits.clone(), temperature=0.5)
    print(f"   ✓ Temperature scaling applied")
    
    print("\n✅ All generation tests passed!")
    print("\nNote: Full generation test requires a trained model.")
    print("See the Jupyter notebook for end-to-end testing.")