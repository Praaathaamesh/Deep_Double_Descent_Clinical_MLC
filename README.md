# Deep Double Descent Demonstration in Clincal MLC (Multi-label classification) Problem

## Everything About Deep Double Descent
- Increase in the model complexity initially decreases the performance; later increases it.
- Deep Double Descent is a function of both model complexity and number of epochs.
- Increase in the number of training datapoints actually hurts the performance on the evaluation set.
- Undermines the conventional idea of **Bias-Variance Tradeoff** -- since it also undermines the notion of "larger models (in this case deep learning networks) are worse than the simpler ones".

## Demonstrative Criteria
- Using clincal time series ECGs from PTBXL dataset, this assignment demonstrates three major distinctions in the terms of deep double descent, which oppose the conventional ideas.
- Model-wise Deep Double Descent, model complexity is the function of variance
- Epoch-wise Deep Double Descent, increasing epochs prevents overfitting, since number of epochs are function of variance.
- Sample-wise non-monotonicity demonstrates that elevated dataset size affects the model performance in the negative manner.

## References
- Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/kfzx-aw45
- Preetum Nakkiran, Kaplun, G., Bansal, Y., Yang, T., Barak, B., & Ilya Sutskever. (2019). Deep Double Descent: Where Bigger Models and More Data Hurt. https://doi.org/10.48550/arxiv.1912.02292

## Notes
> [!WARNING]
> Contents of this repository will be refactored till the completion of the assignment.
