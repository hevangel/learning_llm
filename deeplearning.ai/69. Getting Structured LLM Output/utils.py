import os
from dotenv import load_dotenv, find_dotenv
from typing import TYPE_CHECKING, Optional, Union, List, Literal, Dict, Any
import numpy as np
import torch
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from PIL import Image

if TYPE_CHECKING:
    from outlines.generate import Generator

from outlines.processors.base_logits_processor import OutlinesLogitsProcessor, Array

# Try importing pandas, but don't fail if not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = Any  # For type hints when pandas is not available

def load_env():
    _ = load_dotenv(find_dotenv())

def get_together_api_key():
    load_env()
    together_api_key = os.getenv("TOGETHER_API_KEY")
    return together_api_key

# Simple class to track token usage.
class UsageTracker:
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    total_tokens: list[int] = []

    def track(self, usage):
        self.input_tokens.append(usage.prompt_tokens)
        self.output_tokens.append(usage.completion_tokens)
        self.total_tokens.append(usage.total_tokens)

    def clear(self):
        self.input_tokens.clear()
        self.output_tokens.clear()
        self.total_tokens.clear()

    def __str__(self):
        return f"Inputs: {self.input_tokens}\nOutputs: {self.output_tokens}"

class LogitTrackingProcessor(OutlinesLogitsProcessor):
    """Tracks logits for both structured and unstructured token generation.
    
    For each position in the sequence, stores:
    - unstructured_logits: Raw logits from the model
    - structured_logits: Logits after applying constraints
    - vocab_tokens: Mapping from vocab indices to token strings
    
    Each logit matrix has:
    - Columns: One for each position in the generated sequence
    - Rows: One for each token in the vocabulary
    """
    
    def __init__(self, processor=None):
        self.processor = processor
        self.unstructured_logits = []  # List of logit arrays, one per position
        self.structured_logits = []    # List of logit arrays, one per position
        self.vocab_tokens = None      # Will store the vocabulary mapping
        self.chosen_tokens = []       # Track actual chosen tokens during generation
        
    def process_logits(self, input_ids: Array, logits: Array) -> Array:
        # Always store the raw logits as unstructured
        self.unstructured_logits.append(logits[0].detach().cpu().numpy().copy())
        
        # Store the actual chosen token ID if available
        if len(input_ids[0]) > 0:
            self.chosen_tokens.append(input_ids[0][-1].item())
        
        # Apply structural constraints if we have a processor
        if self.processor is not None:
            processed = self.processor.process_logits(input_ids, logits)
            self.structured_logits.append(processed[0].detach().cpu().numpy().copy())
            return processed
            
        # For unconstrained generation, structured = unstructured
        self.structured_logits.append(logits[0].detach().cpu().numpy().copy())
        return logits
            
    def get_probabilities(self, as_matrix: bool = False) -> Dict[str, Union[List[NDArray], NDArray]]:
        # Convert logits to probabilities
        unstructured_probs = [
            torch.softmax(torch.tensor(logits), dim=-1).numpy()
            for logits in self.unstructured_logits
        ]
        structured_probs = [
            torch.softmax(torch.tensor(logits), dim=-1).numpy()
            for logits in self.structured_logits
        ]
        
        if as_matrix:
            # Stack arrays into matrices
            unstructured = np.column_stack(unstructured_probs)
            structured = np.column_stack(structured_probs)
        else:
            # Return as lists
            unstructured = unstructured_probs
            structured = structured_probs
            
        return {
            'unstructured': unstructured,
            'structured': structured
        }

    def get_logits(self, as_matrix: bool = False) -> Dict[str, Union[List[NDArray], NDArray]]:
        if as_matrix:
            unstructured = np.column_stack(self.unstructured_logits)
            structured = np.column_stack(self.structured_logits)
        else:
            unstructured = self.unstructured_logits
            structured = self.structured_logits
            
        return {
            'unstructured': unstructured,
            'structured': structured
        }
        
    def get_top_tokens(
        self,
        k: int = 10,
        positions: Optional[Union[int, List[int]]] = None,
        include_logits: bool = True
    ) -> List[Dict[str, Any]]:
        if positions is None:
            positions = list(range(len(self.structured_logits)))
        elif isinstance(positions, int):
            positions = [positions]
            
        probs = self.get_probabilities()
        logits = self.get_logits() if include_logits else None
        vocab = self.get_vocab_mapping()
        
        results = []
        for pos in positions:
            if pos >= len(self.unstructured_logits):
                continue
                
            text_so_far = self.sequence(pos)
            
            u_probs = probs['unstructured'][pos]
            s_probs = probs['structured'][pos]
            
            if include_logits:
                u_logits = logits['unstructured'][pos]
                s_logits = logits['structured'][pos]
            
            top_indices = np.argsort(np.maximum(u_probs, s_probs))[-k:][::-1]
            next_token = self.sequence(pos + 1)[len(text_so_far):] if pos < len(self.structured_logits)-1 else ""
            
            tokens = []
            for idx in top_indices:
                token = vocab[idx]
                token_info = {
                    'token': token,
                    'natural_prob': float(u_probs[idx]),
                    'constrained_prob': float(s_probs[idx]),
                    'is_chosen': token == next_token
                }
                
                if include_logits:
                    token_info.update({
                        'natural_logit': float(u_logits[idx]),
                        'constrained_logit': float(s_logits[idx])
                    })
                    
                tokens.append(token_info)
            
            results.append({
                'position': pos,
                'text_so_far': text_so_far,
                'tokens': tokens
            })
            
        return results

    def get_vocab_mapping(self) -> List[str]:
        if not hasattr(self, 'tokenizer'):
            raise AttributeError("No tokenizer available for mapping tokens")
            
        if self.vocab_tokens is None:
            self.vocab_tokens = [
                self.processor.tokenizer.decode([i])[0]
                for i in range(len(self.unstructured_logits[0]))
            ]
            
        return self.vocab_tokens
        
    def clear(self):
        self.unstructured_logits = []
        self.structured_logits = []
        self.chosen_tokens = []

    def to_dataframe(
        self,
        show: Literal["probs", "logits"] = "probs",
        top_k: Optional[int] = None,
        min_value: Optional[float] = None
    ) -> "pd.DataFrame":
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for DataFrame support. "
                "Please install it with: pip install pandas"
            )
            
        if show == "probs":
            values = self.get_probabilities()
        else:
            values = self.get_logits()
            
        vocab = self.get_vocab_mapping()
        rows = []
        
        for pos in range(len(self.unstructured_logits)):
            u_vals = values['unstructured'][pos]
            s_vals = values['structured'][pos]
            
            if top_k is not None or min_value is not None:
                max_vals = np.maximum(u_vals, s_vals)
                
                if top_k is not None and min_value is not None:
                    valid_indices = np.where(max_vals >= min_value)[0]
                    if len(valid_indices) > top_k:
                        valid_indices = valid_indices[np.argsort(max_vals[valid_indices])[-top_k:]]
                elif top_k is not None:
                    valid_indices = np.argsort(max_vals)[-top_k:]
                else:  # min_value is not None
                    valid_indices = np.where(max_vals >= min_value)[0]
            else:
                valid_indices = range(len(vocab))
            
            for idx in valid_indices:
                rows.append({
                    'position': pos,
                    'token': vocab[idx],
                    'natural': u_vals[idx],
                    'constrained': s_vals[idx]
                })
        
        return pd.DataFrame(rows)

    def sequence(self, pos: Optional[int] = None) -> str:
        if not self.chosen_tokens:
            return ""
            
        if not hasattr(self, 'tokenizer'):
            raise AttributeError("No tokenizer available for decoding sequence")
            
        if hasattr(self.processor, 'tokenizer'):
            tokenizer = self.processor.tokenizer
        else:
            tokenizer = self.tokenizer
            
        end_pos = len(self.chosen_tokens) if pos is None else pos
        tokens_to_decode = self.chosen_tokens[:end_pos]
        
        return "".join(tokenizer.decode(tokens_to_decode))

def track_logits(generator: "Generator") -> "Generator":
    """Add probability tracking to any generator."""
    if generator.logits_processor is None:
        raise ValueError("Logit tracking is not supported for this generator")

    tracking = LogitTrackingProcessor(generator.logits_processor)

    if hasattr(generator.logits_processor, 'tokenizer'):
        tracking.tokenizer = generator.logits_processor.tokenizer
    
    generator.logits_processor = tracking
    
    return generator

# Initialize tokenizer
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def template(prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, 
         {"role": "user", "content": prompt}],
        tokenize=False,
        add_bos=True,
        add_generation_prompt=True,
    )

def load_and_resize_image(image_path, max_size=1024):
    """Load and resize an image while maintaining aspect ratio"""
    image = Image.open(image_path)
    width, height = image.size
    scale = min(max_size / width, max_size / height)
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

DEFAULT_BASE_PROMPT="Is this a hotdog or not a hotdog"
def get_messages(image, base_prompt=DEFAULT_BASE_PROMPT):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": base_prompt
                },
            ],
        }
    ]
    return messages

def plot_token_distributions(tracking_processor, k=10, positions=None, prefix=""):
    """Plot token probability distributions before and after applying constraints."""
    probs = tracking_processor.get_probabilities(as_matrix=True)
    vocab = tracking_processor.get_vocab_mapping()
    
    if positions is None:
        positions = list(range(probs['unstructured'].shape[1]))
    n_positions = len(positions)
    
    fig, axes = plt.subplots(1, n_positions)
    if n_positions == 1:
        axes = [axes]
    
    for idx, pos in enumerate(positions):
        unstructured = probs['unstructured'][:, pos]
        structured = probs['structured'][:, pos]
        
        top_indices = np.argsort(np.maximum(unstructured, structured))[-k:]
        
        y = np.arange(len(top_indices))
        height = 0.35
        
        axes[idx].barh(y - height/2, unstructured[top_indices], height, 
                      label='Unconstrained', alpha=0.7, color='skyblue')
        axes[idx].barh(y + height/2, structured[top_indices], height,
                      label='Constrained', alpha=0.7, color='orange')
        
        axes[idx].set_title('Next token probability')
        axes[idx].set_yticks(y)
        axes[idx].set_yticklabels([vocab[i] for i in top_indices])
        axes[idx].set_xlabel('Probability')
        axes[idx].tick_params(axis='both', labelsize=16)
        
        axes[idx].legend(loc='lower right', bbox_to_anchor=(1, 1.1))
        axes[idx].grid(True, alpha=0.3)
        
        for i, (v1, v2) in enumerate(zip(unstructured[top_indices], structured[top_indices])):
            if v1 > 0.01:
                axes[idx].text(v1 + 0.01, i - height/2, f'{v1:.1%}', va='center')
            if v2 > 0.01:
                axes[idx].text(v2 + 0.01, i + height/2, f'{v2:.1%}', va='center')
    
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_heatmap(tracking_processor, k=50, positions=None, prefix="", show_both=True, kind="logits", show_tokens=True):
    """Plot a heatmap of token probabilities across sequence positions."""
    if kind == "logits":
        things = tracking_processor.get_logits(as_matrix=True)
        threshold = -1e9
    else:
        things = tracking_processor.get_probabilities(as_matrix=True)
        threshold = 0.001
    
    vocab = tracking_processor.get_vocab_mapping()
    
    if positions is None:
        positions = list(range(things['unstructured'].shape[1]))
    
    max_probs = np.maximum(
        things['unstructured'].max(axis=1),
        things['structured'].max(axis=1)
    )
    top_indices = np.argsort(max_probs)[-k:]
    
    def mask_array(arr):
        if kind == "logits":
            return np.ma.masked_where(arr < threshold, arr)
        else:
            return np.ma.masked_where(arr < threshold, arr)
    
    unstructured_masked = mask_array(things['unstructured'][top_indices][:, positions])
    structured_masked = mask_array(things['structured'][top_indices][:, positions])

    unstructured_masked, structured_masked = [(x - x.mean(0)) / x.std(0) for x in (unstructured_masked, structured_masked)]

    if show_both:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
        fig.suptitle(f'Token {kind.capitalize()} Evolution', fontsize=16, y=1.05)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    
    im1 = ax1.imshow(
        unstructured_masked,
        aspect='auto',
        cmap='viridis',
    )
    ax1.set_title(f'Natural Token {kind.capitalize()}')
    ax1.set_xlabel('Position in Sequence')
    ax1.set_ylabel('Token')
    if show_tokens:
        ax1.set_yticks(range(len(top_indices)))
        ax1.set_yticklabels([vocab[i] for i in top_indices])
    plt.colorbar(im1, ax=ax1, label=f'{kind.capitalize()}')
    
    if show_both:
        im2 = ax2.imshow(
            structured_masked,
            aspect='auto',
            cmap='viridis',
        )
        ax2.set_title(f'Constrained Token {kind.capitalize()}')
        ax2.set_xlabel('Position in Sequence')
        ax2.set_yticks([])
        plt.colorbar(im2, ax=ax2, label=f'{kind.capitalize()}')
    
    plt.tight_layout()
    plt.show()
    plt.close()
