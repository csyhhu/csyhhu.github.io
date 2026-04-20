---
layout: post
title: MixFormer - Co-Scaling Dense and Sequence Features in Industrial Recommenders
---

# Introduction

# Key Compoenet

## SwishGLU

### Comparison with Other Activation Functions

| Aspect | ReLU | GELU | Swish | SwiGLU |
|--------|------|------|-------|--------|
| **Formula** | max(0, x) | x · Φ(x) | x · sigmoid(x) | (Swish(xW) ⊗ xV) W_out |
| **Smoothness** | Non-smooth (hard cutoff) | Smooth | Smooth | Smooth with gating |
| **Gradient Flow** | Can have dead neurons | Good | Good | Excellent (gated) |
| **Parameters** | None | None | None | Yes (W, V, W_out) |
| **Gating Mechanism** | No | No | Self-gating | Learnable gating (xV) |
| **Computational Cost** | Very Low | Low | Low | Medium |
| **Expressiveness** | Limited | Moderate | Moderate | High (gate learns importance) |
| **Use in Large Models** | Legacy (older) | Common (BERT, GPT-2) | Emerging | State-of-the-art (PaLM, GPT-3+) |
| **Best For** | Simple networks | General purpose | Smooth non-linearity | Complex feature interactions |

### Key Advantages of SwiGLU

1. **Learnable Gate**: The gate `xV` adapts to learn which feature combinations matter most
2. **Smooth Gradients**: Unlike ReLU, no gradient clipping or dead neuron issues
3. **Expressive Power**: Significantly outperforms simpler activations in capturing non-linear patterns
4. **Empirical Performance**: Proven effectiveness in large-scale models (PaLM, ChatGPT series)

## Per-Head FFN

Per-Head FFN means **each attention head has its own independent Feed-Forward Network** instead of sharing a single projection layer.

**Standard Multi-Head Attention**:
```
[Head 1, Head 2, ..., Head h] → Concatenate → Shared Linear Projection → Output
```

**Per-Head FFN Architecture**:
```
[Head 1 + FFN₁] 
[Head 2 + FFN₂] 
...
[Head h + FFNₕ] → Concatenate → Output
```

### Parameter Count Analysis

Assume:
- Model dimension: `d_model = 512`
- Number of heads: `h = 8`
- Head dimension: `d_head = 64`
- FFN hidden dimension: `d_ff = 256`

**Standard Projection**:
- Parameters: 512 × 512 = 262,144

**Per-Head FFN** (8 heads):
- Per head: (64 × 256) + (256 × 64) = 33,088 parameters
- Total: 33,088 × 8 = 264,704 parameters

| Method | Parameters | Comparison |
|--------|-----------|-----------|
| Standard Projection | 262,144 | Baseline |
| Per-Head FFN | 264,704 | ↑ 0.98% (negligible) |

### Computational Cost Analysis

**Theoretical FLOPs**:

Standard Projection:
```
batch × seq_len × 512 × 512 × 2 = batch × seq_len × 524,288 FLOPs
```

Per-Head FFN (all heads):
```
batch × seq_len × (64×256 + 256×64) × 2 × 8 = batch × seq_len × 524,288 FLOPs
```

**Result**: Computational complexity is **essentially identical**

### Practical Considerations

| Aspect | Standard Projection | Per-Head FFN |
|--------|-------------------|-------------|
| **Parameters** | ✅ 262K | ✅ 265K (~same) |
| **Theoretical FLOPs** | ✅ Same | ✅ Same |
| **Hardware Efficiency** | ⚠️ One large matrix mult | ⚠️ Multiple small matrix mults |
| **Memory Footprint** | Depends | Depends on implementation |
| **Expressiveness** | Limited (shared projection) | ✅ Higher (independent FFNs) |

### Why Use Per-Head FFN?

Despite similar computational cost, Per-Head FFN offers advantages:

1. **Higher Expressiveness**: Each head can learn its own non-linear transformation specific to the patterns it captures
2. **Reduced Information Loss**: Standard shared projection may lose head-specific information
3. **Better Generalization**: Independent transformations increase model diversity and reduce overfitting
4. **Flexible Capacity**: Each head's FFN can adapt to its unique role in the attention mechanism

### Summary

Per-Head FFN is a **parameter-efficient** architecture choice that provides:
- ✅ Minimal parameter overhead (<1%)
- ✅ Similar computational cost
- ✅ Significantly improved expressiveness
- ✅ Better learning capacity for complex feature interactions