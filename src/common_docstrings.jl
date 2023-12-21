# commonly used DocStrings
const VERBOSE = "`verbose` [Default: `false`, Type: `Bool`]: print information about how the NSSS is solved (symbolic or numeric), which solver is used (Levenberg-Marquardt...), and the maximum absolute error."
const MODEL = "`𝓂`: the object created by [`@model`](@ref) and [`@parameters`](@ref) for which to get the solution."
const PARAMETER_VALUES = "`parameters` [Type: `Vector`]: Parameter values in alphabetical order (sorted by parameter name)."
const PARAMETERS = "`parameters` [Default: `nothing`]: If nothing is provided, the solution is calculated for the parameters defined previously. Acceptable inputs are a vector of parameter values, a vector or tuple of pairs of the parameter `Symbol` or `String` and value. If the new parameter values differ from the previously defined the solution will be recalculated."
const VARIABLES = "`variables` [Default: `:all_excluding_obc`]: variables for which to show the results. Inputs can be a variable name passed on as either a `Symbol` or `String` (e.g. `:y` or \"y\"), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. Any variables not part of the model will trigger a warning. `:all_excluding_auxilliary_and_obc` contains all shocks less those related to auxilliary variables and related to occasionally binding constraints (obc). `:all_excluding_obc` contains all shocks less those related to auxilliary variables. `:all` will contain all variables."
const SHOCKS = "`shocks` [Default: `:all_excluding_obc`]: shocks for which to calculate the IRFs. Inputs can be a shock name passed on as either a `Symbol` or `String` (e.g. `:y`, or \"y\"), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. `:simulate` triggers random draws of all shocks (excluding occasionally binding constraints (obc) related shocks). `:all_excluding_obc` will contain all shocks but not the obc related ones.`:all` will contain also the obc related shocks. A series of shocks can be passed on using either a `Matrix{Float64}`, or a `KeyedArray{Float64}` as input with shocks (`Symbol` or `String`) in rows and periods in columns. The period of the simulation will correspond to the length of the input in the period dimension + the number of periods defined in `periods`. If the series of shocks is input as a `KeyedArray{Float64}` make sure to name the rows with valid shock names of type `Symbol`. Any shocks not part of the model will trigger a warning. `:none` in combination with an `initial_state` can be used for deterministic simulations."
const DERIVATIVES = "`derivatives` [Default: `true`, Type: `Bool`]: calculate derivatives with respect to the parameters."
const PERIODS = "`periods` [Default: `40`, Type: `Int`]: number of periods for which to calculate the IRFs. In case a matrix of shocks was provided, periods defines how many periods after the series of shocks the simulation continues."
const NEGATIVE_SHOCK = "`negative_shock` [Default: `false`, Type: `Bool`]: calculate a negative shock. Relevant for generalised IRFs."
const GENERALISED_IRF = "`generalised_irf` [Default: `false`, Type: `Bool`]: calculate generalised IRFs. Relevant for nonlinear solutions. Reference steady state for deviations is the stochastic steady state."
const INITIAL_STATE = "`initial_state` [Default: `[0.0]`, Type: `Vector{Float64}`]: provide state (in levels, not deviations) from which to start IRFs. Relevant for normal IRFs. The state includes all variables as well as exogenous variables in leads or lags if present."
const ALGORITHM = "`algorithm` [Default: `:first_order`, Type: `Symbol`]: algorithm to solve for the dynamics of the model."
const LEVELS = "`levels` [Default: `false`, Type: `Bool`]: return levels or absolute deviations from steady state corresponding to the solution algorithm (e.g. stochastic steady state for higher order solution algorithms)."
const CONDITIONS = "`conditions` [Type: `Union{Matrix{Union{Nothing,Float64}}, SparseMatrixCSC{Float64}, KeyedArray{Union{Nothing,Float64}}, KeyedArray{Float64}}`]: conditions for which to find the corresponding shocks. The input can have multiple formats, but for all types of entries the first dimension corresponds to the number of variables and the second dimension to the number of periods. The conditions can be specified using a matrix of type `Matrix{Union{Nothing,Float64}}`. In this case the conditions are matrix elements of type `Float64` and all remaining (free) entries are `nothing`. You can also use a `SparseMatrixCSC{Float64}` as input. In this case only non-zero elements are taken as conditions. Note that you cannot condition variables to be zero using a `SparseMatrixCSC{Float64}` as input (use other input formats to do so). Another possibility to input conditions is by using a `KeyedArray`. You can use a `KeyedArray{Union{Nothing,Float64}}` where, similar to `Matrix{Union{Nothing,Float64}}`, all entries of type `Float64` are recognised as conditions and all other entries have to be `nothing`. Furthermore, you can specify in the primary axis a subset of variables (of type `Symbol` or `String`) for which you specify conditions and all other variables are considered free. The same goes for the case when you use `KeyedArray{Float64}}` as input, whereas in this case the conditions for the specified variables bind for all periods specified in the `KeyedArray`, because there are no `nothing` entries permitted with this type."
const SHOCK_CONDITIONS = "`shocks` [Default: `nothing`, Type: `Union{Matrix{Union{Nothing,Float64}}, SparseMatrixCSC{Float64}, KeyedArray{Union{Nothing,Float64}}, KeyedArray{Float64}, Nothing} = nothing`]: known values of shocks. This entry allows the user to include certain shock values. By entering restrictions on the shock sin this way the problem to match the conditions on endogenous variables is restricted to the remaining free shocks in the repective period. The input can have multiple formats, but for all types of entries the first dimension corresponds to the number of shocks and the second dimension to the number of periods. The shocks can be specified using a matrix of type `Matrix{Union{Nothing,Float64}}`. In this case the shocks are matrix elements of type `Float64` and all remaining (free) entries are `nothing`. You can also use a `SparseMatrixCSC{Float64}` as input. In this case only non-zero elements are taken as certain shock values. Note that you cannot condition shocks to be zero using a `SparseMatrixCSC{Float64}` as input (use other input formats to do so). Another possibility to input known shocks is by using a `KeyedArray`. You can use a `KeyedArray{Union{Nothing,Float64}}` where, similar to `Matrix{Union{Nothing,Float64}}`, all entries of type `Float64` are recognised as known shocks and all other entries have to be `nothing`. Furthermore, you can specify in the primary axis a subset of shocks (of type `Symbol` or `String`) for which you specify values and all other shocks are considered free. The same goes for the case when you use `KeyedArray{Float64}}` as input, whereas in this case the values for the specified shocks bind for all periods specified in the `KeyedArray`, because there are no `nothing` entries permitted with this type."
const PARAMETER_DERIVATIVES = "`parameter_derivatives` [Default: :all]: parameters for which to calculate partial derivatives. Inputs can be a parameter name passed on as either a `Symbol` or `String` (e.g. `:alpha`, or \"alpha\"), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. `:all` will include all parameters."
const DATA = "`data` [Type: `KeyedArray`]: data matrix with variables (`String` or `Symbol`) in rows and time in columns"
