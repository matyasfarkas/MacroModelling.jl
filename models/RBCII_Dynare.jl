# RBC-II Model from Dynare SEP simulation example (rbcii.mod)
# Source: /Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE/ep-mj-30-years-master/models/rbcii/rbcii.mod

using MacroModelling

@model RBCII_Dynare begin
    # Logged TFP (AR(1) process)
    efficiency[0] = rho * efficiency[-1] + sigma * epsilon[x]

    # TFP level
    Efficiency[0] = Effstar * exp(efficiency[0])

    # Production function (CES)
    Output[0] = Efficiency[0] * (alpha * Capital[-1]^psi + (1 - alpha) * Labour[0]^psi)^(1 / psi)

    # Capital law of motion
    Capital[0] = Output[0] - Consumption[0] + (1 - delta) * Capital[-1]

    # Consumption/Leisure arbitrage (intratemporal)
    (1 - theta) / theta * Consumption[0] / (1 - Labour[0]) = (1 - alpha) * (Output[0] / Labour[0])^(1 - psi)

    # Euler equation (with occasionally binding constraint)
    (Consumption[0]^theta * (1 - Labour[0])^(1 - theta))^(1 - tau) / Consumption[0] - LagrangeMultiplier[0] =
        beta * ((Consumption[1]^theta * (1 - Labour[1])^(1 - theta))^(1 - tau) / Consumption[1] *
        (alpha * (Output[1] / Capital[0])^(1 - psi) + 1 - delta) + LagrangeMultiplier[1] * (1 - delta))

    # Investment (accounting identity)
    Investment[0] = Output[0] - Consumption[0]

    # Complementarity: LagrangeMultiplier >= 0, Investment >= ZLB * Investment[ss]
    min(LagrangeMultiplier[0], Investment[0] - ZLB * Investment[ss]) = 0
end

@parameters RBCII_Dynare begin
    # Technology
    Effstar = 1.000
    rho = 0.950
    sigma = 0.007

    # Production
    alpha = 0.450
    psi = -0.200

    # Preferences
    beta = 0.990
    theta = 0.357
    tau = 2.000

    # Depreciation
    delta = 0.025

    # Investment floor (fraction of steady state)
    ZLB = 0.85
end
