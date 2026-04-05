## Evidence pile ()

NOTE: Models that were trained without positional embeddings will be referred to as "ablated models" and models trained with them will be referred to as "intact models"

- When observing the ablated models' tokens' cosine similarities, we notice a diagonal + grid-like pattern, similar to the ones we observe in the intact models (but weaker). This pattern, which gets stronger as the tokens pass through the layers of the models, suggests the existence of a spatial structure behind the attention mechanisms of these models that do not have Positional Encodings.

- Models that are trained with Positional Encodings and have them removed during inference, still exhibit this diagonal pattern in the token cosine similarities.

- Ablated models that were trained with random patch permutation (RPP), and then had that patch permutation removed at inference perform similarly and have similar robustness to ablated models trained without patch permutation. On top of that, they (ablated models trained with RPP that is removed post-hoc) exhibit that diagonal pattern in their token similarities. This suggests a statistical/content-based approach to spatial structure.

- Ablated models consistently exhibit higher token cosine similarities that explode even higher and become more uniform on the shifted datasets (AdaIN stylization and Gaussian Blur). This suggests the patch-statistical-based strategy to spatial structure collapses when all the patches become more uniform in terms of content, which explains part of the loss in robustness.

- Models trained with positional embeddings consistently exhibit significantly higher residual stream dimensionality than models trained without. Models trained with positional embeddings but have them removed during inference exhibit dimensionality similar to that of a model never trained with positional embeddings, suggesting that PEs inject this high-dimensionality into the residual stream at inference.

- When evaluating on shifted datasets, these models seem to consistently over-predict certain well-defined classes and under-predict others (e.g. in the CIFAR-10 dataset, the models consistently over-predict the cat class and under-predict the automobile class). This class-specific collapse is significantly heightened when positional embeddings are removed.