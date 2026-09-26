# Implementation Document

This document summarizes the current implementation status for the Algorithms and AI project repository.

## 1. Program structure

The repository is organized into small modules:

- `include/vadugrad/core` and `src/`:
  - tensor representation and low-level tensor helpers (`flatten_bt`, `split_heads`, etc.)
- `include/vadugrad/nn` and `src/`:
  - neural-network components (`DenseLinear`, `MultiHeadAttention`, `LayerNorm`, `FeedForward`, `Embedding`, `TransformerBlock`, `DecoderOnlyTransformer`)
  - activation and loss-related operations
- `include/vadugrad/optim` and `src/`:
  - optimizer (`Adam`)
- `apps/`:
  - runnable demos (`train_lm`)
- `tests/`:
  - GoogleTest unit and integration tests

High-level data flow for language-model training demo:

1. Token ids -> token embeddings + positional embeddings
2. Repeated transformer blocks (pre-norm, MHA, residual, pre-norm, FFN, residual)
3. Final layer norm -> LM head -> logits
4. Cross-entropy loss
5. Manual backward pass for all modules
6. Adam parameter update

## 2. Complexity overview

Let:

- `B` = batch size
- `T` = sequence length
- `D` = model dimension
- `H` = number of heads
- `Dh = D/H`
- `F` = feed-forward hidden dimension
- `L` = number of transformer blocks

### Multi-head attention (single block)

- Q/K/V projections: `O(B*T*D^2)`
- Attention score computation and weighted value mix: `O(B*H*T*T*Dh)` = `O(B*T*T*D)`
- Output projection: `O(B*T*D^2)`

Dominant term depends on `T` vs `D`:

- long sequence: `O(B*T*T*D)`
- wide model: `O(B*T*D^2)`

### Feed-forward network (single block)

- `D -> F -> D`: `O(B*T*D*F)`

### Total per forward pass

For `L` blocks:

- `O(L * (B*T*D^2 + B*T*T*D + B*T*D*F))`

Backward pass is same order of magnitude (constant factors larger).

### Space

- Parameters:
  - embeddings: `O(V*D + T_max*D)`
  - each block: `O(D^2 + D*F)`
- Activations/cache for backward:
  - roughly `O(L*B*T*D)` + attention caches `O(L*B*H*T*T)`

## 3. Performance notes

- Current implementation is intentionally explicit and readable (manual loops over tensor indices).
- This improves reviewability and debugging, but is not optimized for high throughput.
- Main bottlenecks:
  - repeated temporary tensor allocations
  - `T^2` attention loops
  - no SIMD/BLAS acceleration

## 4. Known limitations and improvement ideas

- No dropout yet (train-time regularization missing).
- No mixed precision or hardware acceleration.
- No production dataloader/tokenizer pipeline in core implementation.
- Gradient checking (finite differences) could be expanded for deeper numerical validation.
- Better parameter grouping API could simplify optimizer integration further.

Planned improvements:

1. Add dropout modules and masks
2. Add text corpus loading/tokenization utility
3. Add optional faster tensor kernels for common shapes
4. Add lightweight checkpoint save/load

## 5. Use of large language models

Large language model assistance was used during development for:

- API design iteration
- boilerplate generation for repetitive module/test scaffolding
- code-review style feedback on shape contracts and error handling
- documentation drafting

The implementation, adaptation to this repository, debugging, and validation were performed in this codebase with local build/test execution.

## 6. Sources used

- Vaswani et al., *Attention Is All You Need* (NeurIPS 2017)
- Goodfellow, Bengio, Courville, *Deep Learning* (MIT Press)
- Official GoogleTest documentation
- C++17 standard library references for containers/math/utilities
