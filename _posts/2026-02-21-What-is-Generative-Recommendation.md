---
layout: post
title: What is Generative Recommendation
---

# Introduction

Generative Recommendation (GR) has gained significant popularity recently. Top-tier companies claim that they have successfully converted traditional Deep Learning Recommendation Systems (DLRM) to GR. It appears that recommendation systems (RS) without generative capabilities may become outdated.

The definition of GR is straightforward: generate the exposed recommended item (as Large Language Models (LLM) do) instead of ranking candidates and selecting the top-ranked ones, where ranking is based on the probability of item interaction (click, purchase).

However, a question arises: when LLM generates text, it seems to do something similar - it selects tokens with the highest probability for exposure. Why isn't DLRM considered generative?

# LLM's Generation

Unlike traditional recommendation systems that simply rank items and select the top-1, LLMs use sophisticated sampling strategies for text generation. The key difference is that LLMs don't always choose the most probable token - they introduce controlled randomness through temperature and nucleus (top-p) sampling to create diverse and natural text.

## The Generation Process

At each step of generation, the LLM produces a probability distribution over the entire vocabulary:

```
P(token_i | context) = softmax(logits_i)
```

The generation strategy determines which token to select from this distribution.

## Temperature Scaling

Temperature (τ) controls the "flatness" of the probability distribution:

- **τ = 1.0**: Original distribution (no scaling)
- **τ < 1.0**: Sharpens distribution (more deterministic, focuses on high-probability tokens)
- **τ > 1.0**: Flattens distribution (more random, allows low-probability tokens)

The temperature-scaled logits are computed as:

```python
def temperature_scale(logits, temperature):
    """
    Scale logits by temperature to control randomness
    
    Args:
        logits: Raw output scores from the model [vocab_size]
        temperature: Temperature parameter (0 < τ < ∞)
        
    Returns:
        Temperature-scaled logits
    """
    if temperature == 0:
        raise ValueError("Temperature must be positive")
    
    return logits / temperature
```

## Top-p (Nucleus) Sampling

Top-p sampling keeps the smallest set of highest-probability tokens whose cumulative probability exceeds a threshold p (typically 0.9-0.95). This focuses sampling on the "nucleus" of the distribution while maintaining diversity.

**Why it's important:**
- Removes unlikely tokens that would degrade quality
- Keeps the number of candidates dynamic based on confidence
- Better than fixed top-k in handling uncertain situations

```python
def top_p_sampling(logits, top_p=0.9):
    """
    Nucleus sampling: keep tokens with cumulative probability ≤ p
    
    Args:
        logits: Raw or temperature-scaled model output [vocab_size]
        top_p: Cumulative probability threshold (default: 0.9)
        
    Returns:
        Sampled token index
    """
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # Compute softmax probabilities
    sorted_probs = softmax(sorted_logits)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens beyond the nucleus
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # Shift removal index to keep the first token that exceeds top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Remove low-probability tokens by setting logit to -inf
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    
    # Re-normalize probabilities over remaining tokens
    filtered_probs = softmax(sorted_logits)
    
    # Sample one token from the filtered distribution
    sampled_index = torch.multinomial(filtered_probs, num_samples=1)
    
    # Map back to original vocabulary indices
    final_token = sorted_indices[sampled_index]
    
    return final_token
```

## Combined Temperature + Top-p Generation

In practice, these two strategies are combined for optimal results:

```python
def generate_with_sampling(model, input_ids, 
                           max_length=100,
                           temperature=0.7,
                           top_p=0.9):
    """
    Generate text using temperature + top-p sampling
    
    Args:
        model: Pre-trained LLM
        input_ids: Tokenized input prompt
        max_length: Maximum tokens to generate
        temperature: Temperature parameter
        top_p: Nucleus sampling threshold
        
    Returns:
        Generated token sequence
    """
    generated_ids = input_ids.clone()
    
    for step in range(max_length):
        # Forward pass to get logits
        outputs = model(generated_ids)
        logits = outputs.logits[:, -1, :]  # Get next token predictions
        
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Apply top-p sampling
        next_token = top_p_sampling(scaled_logits, top_p=top_p)
        
        # Append to sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        # Stop if EOS token is generated
        if next_token == model.config.eos_token_id:
            break
    
    return generated_ids
```

## Practical Settings

```python
# Different scenarios use different sampling parameters:

# Creative writing: high randomness
generation_config = {
    'temperature': 0.8,
    'top_p': 0.95
}

# Technical documentation: more deterministic
generation_config = {
    'temperature': 0.3,
    'top_p': 0.9
}

# Code generation: highly deterministic
generation_config = {
    'temperature': 0.2,
    'top_p': 0.8
}
```

This sampling-based approach is what makes LLMs "generative" - they create novel content through probabilistic selection rather than deterministic ranking.

## DLRM's Ranking v.s. LLM's Generation

| Aspect | Traditional DLRM | LLM Generation |
|--------|------------------|----------------|
| **Selection Method** | Argmax (deterministic) | Sampling (stochastic) |
| **Candidate Set** | Fixed (all items) | Dynamic (top-p nucleus) |
| **Output** | Single best item | Sampled sequence |
| **Diversity** | Low (same output) | High (varied outputs) |
| **Parameter Control** | None | Temperature, top-p |

Though different, both approaches actually perform a similar ranking procedure, but none of us would agree that LLM is deterministic. The key difference between DLRM and GR is not **how it gets the result**, but **what they model**.

Before going deeper, let's review what and how recommendation systems model.

# Objective of Deep Learning Recommendation System

Traditionally, recommendation systems (RS) select and expose the item that a user is most likely to interact with. Normally, we use Click-Through Rate (CTR) as the metric to measure the user's interest in an item. The item with the highest CTR is selected and exposed to the user. Therefore, instead of generating the token that most likely appears as LLM does (modeling the nature of language), RS modifies the appearance nature of exposure, leading to high conversion. Specifically,

## (1) Click-Through Rate (CTR) Prediction
```python
# Modeling the probability of user clicking on an item
P(click = 1 | user, item, context)
```

This is the most fundamental objective where the model learns to predict whether a user will interact with a given item based on user profile, item features, and contextual information.

## (2) Conversion Rate (CVR) Prediction
```python
# Modeling the probability of conversion after click
P(conversion = 1 | click = 1, user, item, context)
```

For e-commerce scenarios, CVR prediction is crucial as it focuses on the actual purchase or conversion events rather than just clicks.

## (3) Next Item Prediction
```python
# Predicting the next item in user's interaction sequence
P(item_{t+1} | item_1, item_2, ..., item_t, user)
```

This objective models the sequential nature of user behavior, where the history of interactions influences future choices.

## (4) Multi-Objective Optimization
```python
# Combining multiple objectives with weighted importance
Loss = α · L_CTR + β · L_CVR + γ · L_Time
```

Modern systems often optimize multiple objectives simultaneously, balancing engagement, conversion, and other business metrics.

# Generative Recommendation

Following the analysis above, it is easy to find that generative recommendation is basically modeling the nature of exposure (the objective it models), leading to its definition of generation.

Unlike traditional ranking-based systems that predict probabilities for fixed candidate items and select top-ranked ones, generative recommendation systems treat the recommendation problem as a generation task where items are "generated" from a learned distribution, similar to how LLMs generate tokens.

The key insight is that rather than ranking existing candidates, the model learns to directly sample/produce item IDs from a probability distribution conditioned on user preferences and context.

## OneRec: A Representative Generative Recommendation System

OneRec is a pioneering work that applies generative modeling principles to recommendation systems. It treats item IDs as discrete tokens in a "vocabulary" and learns to generate appropriate item sequences using autoregressive generation, similar to language modeling.

### Key Innovations of OneRec

#### (1) Item as Vocabulary
OneRec conceptualizes the entire item catalog as a vocabulary where each item ID corresponds to a token, using item embeddings similar to word embeddings in LLMs. This allows the model to leverage techniques from natural language processing and sequence generation.

#### (2) Autoregressive Item Generation
OneRec models the recommendation process as an autoregressive generation task using transformer-based architecture. The model learns to predict next item probabilities based on user history and contextual information, generating items sequentially where each item conditions subsequent generation.

#### (3) Generation with Sampling Strategies
OneRec employs similar sampling strategies as LLMs for item generation, including temperature scaling and nucleus (top-p) sampling. This enables controlled randomness in generation, allowing for diverse and novel recommendations rather than always selecting the most probable items.

#### (4) Training Objective
OneRec is trained using standard language modeling objectives with cross-entropy loss for positive items and optional contrastive loss for negative items. This approach learns the underlying distribution of items given user context.

### Advantages of OneRec's Generative Approach

| Aspect | Traditional DLRM | OneRec (Generative) |
|--------|------------------|---------------------|
| **Candidate Selection** | Pre-defined candidate set | Generates from entire item space |
| **Diversity** | Limited to top candidates | Sampling provides natural diversity |
| **Novelty** | Biased to popular items | Can generate unexpected items |
| **Scalability** | O(N) scoring per user | O(1) generation per user |
| **Cold Start** | Needs item embeddings | Can generate from distribution |

## Comparison with Traditional Ranking

The fundamental difference lies in the modeling philosophy:

```python
# Traditional DLRM: Probability modeling for ranking
def dlrm_objective(model, user, item):
    return log P(click | user, item)  # Learn to predict interactions

# OneRec: Distribution modeling for generation  
def onerec_objective(model, user, context):
    return log P(item | user, context)  # Learn the item distribution
```

While both approaches involve probability distributions, the key distinction is:

- **DLRM**: Models interaction probabilities for existing candidates → ranking
- **OneRec**: Models the underlying item distribution → generation

This paradigm shift enables generative recommendation systems to directly produce recommendations rather than ranking pre-defined candidates, offering greater flexibility and potential for novel discovery.

# Scaling v.s. Generative Recommendation

Most so-called Generative Recommendation is not actually modeling the probability of item occurrence, but still modeling the interaction probability (CTR, CVR). They use a Transformers-based architecture and scaling approach, which is basically a scaling law but not generative.

## Kunlun: Establishing Scaling Laws for Recommendation Systems

Meta's recent work "Kunlun: Establishing Scaling Laws for Massive-Scale Recommendation Systems through Unified Architecture Design" establishes scaling laws for recommendation systems similar to those found in large language models. These laws describe how model performance scales with:

- **Model Size**: Number of parameters in the recommendation model
- **Data Volume**: Amount of training data (user interactions, impressions)
- **Compute Budget**: Computational resources available for training and inference

The key insight is that as we increase these factors, recommendation performance improves predictably according to power-law relationships, rather than diminishing returns that plateau early.

### Performance Optimization through Scaling Laws

Rather than arbitrary architectural choices, Kunlun uses scaling laws to guide model design:

- **Optimal Model Size**: Determining the ideal number of parameters for given constraints
- **Resource Allocation**: Balancing memory, latency, and throughput requirements
- **Data Efficiency**: Understanding how much data is needed to train models of different sizes

This principled approach leads to better performance with more efficient resource utilization.

# Conclusion

Generative Recommendation differs from traditional RS in its modeling objective: it models the exposure sequence instead of exposing the candidates with high interaction probability.
