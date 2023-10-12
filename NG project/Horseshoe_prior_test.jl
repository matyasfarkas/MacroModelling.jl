# Horseshoe prior

using Turing

using FillArrays
using LinearAlgebra

struct HorseShoePrior{T}
    X::T
end
O