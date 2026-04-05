## Main claims

**Terminology.**  
Models trained without positional embeddings are referred to as *ablated models*.  
Models trained with positional embeddings are referred to as *intact models*.

---

### Claim 1: Positional embeddings causally alter representation geometry

Positional embeddings inject high-dimensional, heterogeneous, and explicitly spatially structured representations into the residual stream immediately, before the first encoder block. SSDC remains high even when token content is permuted, indicating that intact models rely both on positional embeddings and patch content for spatial information.

---

### Claim 2: Spatial structure can emerge without positional embeddings, but it is content-driven

Without positional embeddings, ablated models gradually recover spatial structure over the first few encoder blocks, likely via attention-mediated composition of patch content. This allows them to achieve near-intact accuracy on small datasets like CIFAR-10, but reliance on content-based heuristics renders them fragile to distributional shifts.

---

### Claim 3: Robustness depends on position-aligned spatial structure, not content-based heuristics alone

Effective rank and token heterogeneity alone are insufficient to explain robustness. Instead, early, positionally-aligned spatial structure appears necessary for robust generalization; models reconstructing structure from content heuristics are less resilient to distributional shifts.

Positional embeddings contribute to spatial alignment rather than acting solely as a direct encoding of absolute position.

---

## Scope and non-claims

- This work does **not** claim that positional embeddings are sufficient for robustness.
- This work does **not** claim that positional embeddings uniquely determine spatial structure.
- Geometric interpretations (e.g., class manifold collapse or thickening) are treated as explanatory hypotheses rather than primary claims.
