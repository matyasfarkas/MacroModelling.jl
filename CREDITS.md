# Credits

## Original Codebase

This repository is built upon **[MacroModelling.jl](https://github.com/thorek1/MacroModelling.jl)**, created and maintained by **Thore Kockerols**.

MacroModelling.jl provides the foundational framework for DSGE model estimation and analysis that this project extends with neural network surrogate-based estimation techniques for regime-switching models.

### Citation

For the original MacroModelling.jl framework, please cite:
- Kockerols, T. (2024). MacroModelling.jl: A Julia package for DSGE model estimation and simulation.

## Extensions in This Repository

This repository extends MacroModelling.jl with:

- **Neural Network Surrogate Integration**: Efficient surrogate models for likelihood evaluation
- **Regime-Switching Estimation Framework**: Hard-gate regime switching for models with occasionally binding constraints (OBC)
- **Performance Optimizations**: 1.5-2x speedup in estimation procedures
- **Advanced Sampling Methods**: Inversion filter and filter-free sampling techniques

## License

This project maintains the same MIT license as MacroModelling.jl.

## Contributors

- **Thore Kockerols** - Original MacroModelling.jl author and maintainer
- **Matyas Farkas** - Neural network surrogate extensions and regime-switching framework
