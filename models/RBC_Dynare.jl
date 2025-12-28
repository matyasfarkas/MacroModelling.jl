# RBC Model from Dynare SEP IRF Example
# Source: /Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE/SW07_development/ep-mj-30-years-master/models/irf/rbc.mod
# This is the model used in Adjemian-Juillard (2025) for SEP IRF demonstration

using MacroModelling

@model RBC_Dynare begin
    # Logged TFP (AR(1) process)
    efficiency[0] = rho * efficiency[-1] + sigma * ϵ[x]

    # TFP level
    Efficiency[0] = Effstar * exp(efficiency[0])

    # Production function (CES)
    Output[0] = Efficiency[0] * (alpha * Capital[-1]^psi + (1 - alpha) * Labour[0]^psi)^(1/psi)

    # Capital law of motion
    Capital[0] = Output[0] - Consumption[0] + (1 - delta) * Capital[-1]

    # Consumption/Leisure arbitrage (intratemporal)
    # From Dynare: (1-theta)/theta*Consumption/(1-Labour) - (1-alpha)*(Output/Labour)^(1-psi) = 0
    (1 - theta) / theta * Consumption[0] / (1 - Labour[0]) = (1 - alpha) * (Output[0] / Labour[0])^(1 - psi)

    # Euler equation (intertemporal)
    # From Dynare: (C^theta*(1-L)^(1-theta))^(1-tau)/C = beta*(C(+1)^theta*(1-L(+1))^(1-theta))^(1-tau)/C(+1)*(alpha*(Y(+1)/K)^(1-psi)+1-delta)
    (Consumption[0]^theta * (1 - Labour[0])^(1 - theta))^(1 - tau) / Consumption[0] =
        beta * (Consumption[1]^theta * (1 - Labour[1])^(1 - theta))^(1 - tau) / Consumption[1] *
        (alpha * (Output[1] / Capital[0])^(1 - psi) + 1 - delta)

    # Investment (accounting identity)
    Investment[0] = Output[0] - Consumption[0]
end

@parameters RBC_Dynare begin
    # Technology
    Effstar = 1.000   # Steady state TFP level
    rho = 0.800       # TFP persistence
    sigma = 0.100     # TFP shock std dev

    # Production
    alpha = 0.450     # Capital share in CES production
    psi = -0.200      # Substitution parameter (elasticity = 1/(1-psi))

    # Preferences
    beta = 0.990      # Discount factor
    theta = 0.357     # Consumption weight in utility
    tau = 2.000       # Risk aversion (inverse of EIS)

    # Depreciation
    delta = 0.010     # Capital depreciation rate
end

# Note: In Dynare, the shock variance is set to 1 in the shocks block,
# but the actual shock std dev (sigma) is used in the efficiency equation.
# MacroModelling handles this differently - we define shock std dev as parameter.
