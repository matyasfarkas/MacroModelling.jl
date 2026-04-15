"""
IMPROVEMENT 2: Early Stopping for Neural Network Training

**Problem**: NN surrogate training runs for fixed number of epochs, wasting time if convergence
is reached early or overfitting begins.

**Solution**: Implement early stopping with validation set monitoring.

**Expected Impact**: 20-40% reduction in training time with better generalization.

Created: January 2026
Status: Production-ready
"""

using Statistics

"""
    EarlyStoppingState

Tracks validation loss for early stopping decision.

# Fields
- `best_val_loss::Float64`: Best validation loss seen so far
- `best_epoch::Int`: Epoch where best validation loss occurred
- `patience::Int`: How many epochs to wait for improvement
- `min_delta::Float64`: Minimum improvement to count as progress
- `counter::Int`: Epochs since last improvement
- `stopped::Bool`: Whether training was stopped early
"""
mutable struct EarlyStoppingState
    best_val_loss::Float64
    best_epoch::Int
    patience::Int
    min_delta::Float64
    counter::Int
    stopped::Bool
end

"""
    EarlyStoppingState(; patience=20, min_delta=1e-4)

Initialize early stopping state.
"""
function EarlyStoppingState(; patience::Int=20, min_delta::Float64=1e-4)
    return EarlyStoppingState(Inf, 0, patience, min_delta, 0, false)
end

"""
    should_stop!(es::EarlyStoppingState, val_loss::Float64, epoch::Int)

Check if training should stop based on validation loss.

Returns `true` if should stop, `false` otherwise.
"""
function should_stop!(es::EarlyStoppingState, val_loss::Float64, epoch::Int)
    # Check if this is an improvement
    if val_loss < es.best_val_loss - es.min_delta
        # Improvement found
        es.best_val_loss = val_loss
        es.best_epoch = epoch
        es.counter = 0
        return false
    else
        # No improvement
        es.counter += 1
        if es.counter >= es.patience
            es.stopped = true
            return true
        end
        return false
    end
end

"""
    split_train_val(X::Matrix{Float64}, Y::Matrix{Float64}; val_frac::Float64=0.15)

Split data into training and validation sets.

Returns `(X_train, Y_train, X_val, Y_val)`.
"""
function split_train_val(X::Matrix{Float64}, Y::Matrix{Float64}; val_frac::Float64=0.15)
    n = size(X, 2)
    n_val = max(1, round(Int, n * val_frac))
    n_train = n - n_val

    # Random shuffle indices
    indices = randperm(n)
    train_idx = indices[1:n_train]
    val_idx = indices[n_train+1:end]

    X_train = X[:, train_idx]
    Y_train = Y[:, train_idx]
    X_val = X[:, val_idx]
    Y_val = Y[:, val_idx]

    return X_train, Y_train, X_val, Y_val
end

"""
    compute_validation_loss(W1, b1, W2, b2, W3, b3, X_val::Matrix, Y_val::Matrix)

Compute mean squared error on validation set.
"""
function compute_validation_loss(W1, b1, W2, b2, W3, b3, X_val::Matrix, Y_val::Matrix)
    n_val = size(X_val, 2)
    total_loss = 0.0

    for i in 1:n_val
        x = X_val[:, i]
        y_true = Y_val[:, i]

        # Forward pass
        h1 = tanh.(W1 * x .+ b1)
        if W3 === nothing
            y_pred = W2 * h1 .+ b2
        else
            h2 = tanh.(W2 * h1 .+ b2)
            y_pred = W3 * h2 .+ b3
        end

        # MSE
        total_loss += sum((y_pred .- y_true).^2)
    end

    return total_loss / n_val
end

"""
    save_best_params(W1, b1, W2, b2, W3, b3)

Create a deep copy of network parameters.
"""
function save_best_params(W1, b1, W2, b2, W3, b3)
    return (
        W1 = copy(W1),
        b1 = copy(b1),
        W2 = copy(W2),
        b2 = copy(b2),
        W3 = W3 === nothing ? nothing : copy(W3),
        b3 = b3 === nothing ? nothing : copy(b3)
    )
end

"""
    restore_best_params!(W1, b1, W2, b2, W3, b3, best_params)

Restore network parameters from saved best.
"""
function restore_best_params!(W1, b1, W2, b2, W3, b3, best_params)
    W1 .= best_params.W1
    b1 .= best_params.b1
    W2 .= best_params.W2
    b2 .= best_params.b2
    if W3 !== nothing
        W3 .= best_params.W3
        b3 .= best_params.b3
    end
    return nothing
end

"""
    compute_train_val_split_fraction(n::Int)

Compute validation fraction based on dataset size.

Uses adaptive split: smaller datasets get smaller validation sets.
"""
function compute_train_val_split_fraction(n::Int)
    if n < 100
        return 0.20  # 20% validation for very small datasets
    elseif n < 500
        return 0.15  # 15% validation for small datasets
    else
        return 0.10  # 10% validation for larger datasets
    end
end
