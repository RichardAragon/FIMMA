# FIMMA
# Fractal-Inspired Model Merging Algorithm (FIMMA)

The Fractal-Inspired Model Merging Algorithm (FIMMA) is a novel approach to merging multiple deep learning models trained on different domains or tasks. By leveraging fractal geometry and domain-specific knowledge, FIMMA aims to create a unified model that effectively integrates the knowledge and capabilities of the individual models.

## Overview

The key idea behind FIMMA is to represent the weights of each layer in the models as fractal geometries. Fractal representations capture the intricate patterns and self-similarity present in the weights, allowing for a more meaningful comparison and merging of different models.

The algorithm follows an iterative process where it randomly selects two models to merge at each iteration. It identifies layers of the same type (e.g., convolutional, dense) and computes a merge probability based on the similarity of their fractal representations. Layers with higher merge probabilities are then combined using fractal interpolation techniques, taking into account domain-specific rules and constraints.

After merging the selected layers, FIMMA optimizes the fractal dimension of the merged layers to maximize the performance of the resulting model across all domains. This optimization step ensures that the merged model effectively captures and integrates the knowledge from different domains.

The merging process continues for a specified number of iterations, and the best-performing merged model is selected as the final output.

## Key Features

- Fractal representation of layer weights to capture intricate patterns and self-similarity
- Domain-specific interpolation rules for meaningful merging of fractal geometries
- Merge probability calculation based on layer similarity to guide the merging process
- Optimization of fractal dimension to maximize performance across domains
- Iterative merging process to explore different combinations of models and layers
- Evaluation of merged models using domain-specific metrics to assess performance

## Prerequisites

To use FIMMA, you need the following:

- Python 3.x
- NumPy
- SciPy
- Deep learning framework (e.g., TensorFlow, PyTorch) for defining and training models

## Usage

1. Define your deep learning models using the supported layer types (`ConvLayer`, `DenseLayer`).
2. Train the models on different domains or tasks.
3. Specify the domain-specific evaluation metrics (`metric_1`, `metric_2`, `metric_3`) to assess the performance of the merged models.
4. Initialize the models and create a list of models to be merged.
5. Call the `fimma` function with the appropriate parameters:
   - `models`: List of models to be merged
   - `num_iterations`: Number of merging iterations
   - `domain_metrics`: List of domain-specific evaluation metrics
   - `layer_types`: List of supported layer types for merging
6. The `fimma` function will perform the fractal-based model merging and return the best-performing merged model.

## Example

```python
# Initialize TinyLlama models
model_1 = ...  # Initialize model_1
model_2 = ...  # Initialize model_2
model_3 = ...  # Initialize model_3
models = [model_1, model_2, model_3]

# Define domain-specific evaluation metrics
domain_metrics = [metric_1, metric_2, metric_3]

# Specify supported layer types for merging
layer_types = [ConvLayer, DenseLayer]

# Perform fractal-based model merging
merged_model = fimma(models, num_iterations=10, domain_metrics=domain_metrics, layer_types=layer_types)
```

## Customization

FIMMA provides flexibility for customization based on specific requirements and domain knowledge. You can:

- Modify the fractal transformation functions (`perform_conv_fractal_transform`, `perform_dense_fractal_transform`) to incorporate different fractal algorithms or variations.
- Adjust the domain-specific interpolation rules in the `fractal_interpolation` function to align with the characteristics and constraints of your domains.
- Customize the merge probability calculation in the `merge_probability` function to consider different similarity measures or domain-specific factors.
- Adapt the optimization process in the `optimize_alpha` function to use alternative optimization algorithms or incorporate domain-specific constraints.

## Limitations and Future Work

FIMMA is a novel approach to model merging and may have certain limitations:

- The effectiveness of the algorithm depends on the quality and diversity of the individual models being merged.
- The fractal transformations used in the current implementation are based on the Mandelbrot set formula, but other fractal algorithms or variations could be explored for potentially better results.
- The algorithm assumes that the models have compatible architectures and layer types. Merging models with significantly different architectures may require additional adaptations.

Future work on FIMMA could include:

- Extending the algorithm to support a wider range of layer types and architectures.
- Exploring alternative fractal transformations and interpolation techniques for improved merging.
- Incorporating more advanced optimization methods to find the optimal fractal dimensions.
- Conducting extensive experiments and evaluations on various domains and tasks to assess the generalizability and robustness of the algorithm.

## References

- [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set)
- [Fractal geometry](https://en.wikipedia.org/wiki/Fractal_geometry)
- [Model merging techniques](https://arxiv.org/abs/1910.14198)

## Acknowledgments

We would like to thank the contributors and researchers who have inspired and influenced the development of FIMMA.

If you have any questions, suggestions, or feedback, please feel free to contact the project maintainers.

Happy model merging!
