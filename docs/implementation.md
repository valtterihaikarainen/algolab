# Implementation Document

This document summarizes the current implementation status for the Algorithms and AI project repository.

**Peer reviewers:** for build commands, test expectations, MNIST filenames, and a concise code map (butterfly, MNIST, training app), see the root **`README.md`** section **For peer reviewers**.

## 1. Program structure

The repository is organized into small modules:

- `include/vadugrad/core` and `src/`:
  - tensor representation and low-level tensor helpers (`flatten_bt`, `split_heads`, etc.)
- `include/vadugrad/data` and `src/`:
  - MNIST IDX file loading (`mnist.cpp`)
- `include/vadugrad/nn` and `src/`:
  - neural-network components (`DenseLinear`, `ButterflyLinear`, `MultiHeadAttention`, `LayerNorm`, `FeedForward`, `Embedding`, `TransformerBlock`, `DecoderOnlyTransformer`)
  - activation and loss-related operations
- `include/vadugrad/optim` and `src/`:
  - optimizer (`Adam`)
- `apps/`:
  - runnable demos (`train_lm`, `train_mnist`)
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
- `S` = number of butterfly stages (here `S = log2(n)` for `ButterflyLinear`)

### Butterfly linear map (square `n`, block size 2)

Implemented as `ButterflyLinear` with `n` a power of two and `S = log2(n)` stages.

- Forward time (per batch): `O(B * n * S)` = `O(B * n * log n)`
- Parameters: `S * (n/2) * 4` = `O(n log n)`

This is meant as a review-friendly first butterfly building block (fixed `2x2` blocks and a fixed FFT-style pairing schedule).

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
  - `Tensor::operator()` builds a `std::vector<int>` and bounds-checks on each access
  - no SIMD/BLAS acceleration
- `Adam` stores moment buffers keyed by `param.data()`. Reallocating a parameter tensor (new storage) resets those moments.

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

## 5. Version management and language-model assistance

I started this course in spring 2026 and did not finish the project then (course staff can confirm that). The C++ in this repository is that implementation.

This autumn I used the Cursor editor to sort out git between last semester’s local history and the current public remote (week-by-week snapshots, file names, commit messages).

Language-model assistance was also used during development for:

- API design iteration
- boilerplate generation for repetitive module/test scaffolding
- code-review style feedback on shape contracts and error handling
- documentation drafting

The implementation, adaptation to this repository, debugging, and validation were done in this codebase with local build and test runs.

## 6. Sources used

- Vaswani et al., *Attention Is All You Need* (NeurIPS 2017)
- Goodfellow, Bengio, Courville, *Deep Learning* (MIT Press)
- Official GoogleTest documentation
- C++17 standard library references for containers/math/utilities
