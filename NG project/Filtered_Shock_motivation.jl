using MacroModelling

# An Schorfheide model
@model FS2000 begin
           dA[0] = exp(gam + z_e_a  *  e_a[x])
       
           log(m[0]) = (1 - rho) * log(mst)  +  rho * log(m[-1]) + z_e_m  *  e_m[x]
       
           - P[0] / (c[1] * P[1] * m[0]) + bet * P[1] * (alp * exp( - alp * (gam + log(e[1]))) * k[0] ^ (alp - 1) * n[1] ^ (1 - alp) + (1 - del) * exp( - (gam + log(e[1])))) / (c[2] * P[2] * m[1])=0
       
           W[0] = l[0] / n[0]
       
           - (psi / (1 - psi)) * (c[0] * P[0] / (1 - n[0])) + l[0] / n[0] = 0
       
           R[0] = P[0] * (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ ( - alp) / W[0]
       
           1 / (c[0] * P[0]) - bet * P[0] * (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ (1 - alp) / (m[0] * l[0] * c[1] * P[1]) = 0
       
           c[0] + k[0] = exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ (1 - alp) + (1 - del) * exp( - (gam + z_e_a  *  e_a[x])) * k[-1]
       
           P[0] * c[0] = m[0]
       
           m[0] - 1 + d[0] = l[0]
       
           e[0] = exp(z_e_a  *  e_a[x])
       
           y[0] = k[-1] ^ alp * n[0] ^ (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x]))
       
           gy_obs[0] = dA[0] * y[0] / y[-1]
       
           gp_obs[0] = (P[0] / P[-1]) * m[-1] / dA[0]
       
           log_gy_obs[0] = log(gy_obs[0])
       
           log_gp_obs[0] = log(gp_obs[0])
end

@parameters FS2000 begin
        alp     = 0.356
        bet     = 0.993
        gam     = 0.0085
        mst     = 1.0002
        rho     = 0.129
        psi     = 0.65
        del     = 0.01
        z_e_a   = 0.035449
        z_e_m   = 0.008862
end

using CSV, DataFrames, AxisKeys

dat = CSV.read("C:/Users/fm007/Documents/GitHub/MacroModelling.jl/test/data/FS2000_data.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = 1:size(dat)[1])
data = log.(data)
observables = sort(Symbol.("log_".*names(dat)))
data = data(observables,:)
filtered_shocks = get_estimated_shocks(FS2000, data)

using StatsPlots, Distributions
StatsPlots.plot(Distributions.Normal(0,1), fill=(0, .5,:blue), label= "Theoretical - Standard Normal")
        StatsPlots.density!(filtered_shocks[2,:], label = "Empirical -Filtered shocks distribution",title = title = "Monetary policy shock distributions")
        

# import Turing
# import Turing: NUTS, sample, logpdf
# Turing.@model function FS2000_loglikelihood_function(data, m, observables)
#     alp     ~ MacroModelling.Beta(0.356, 0.02, μσ = true)
#     bet     ~ MacroModelling.Beta(0.993, 0.002, μσ = true)
#     gam     ~ MacroModelling.Normal(0.0085, 0.003)
#     mst     ~ MacroModelling.Normal(1.0002, 0.007)
#     rho     ~ MacroModelling.Beta(0.129, 0.223, μσ = true)
#     psi     ~ MacroModelling.Beta(0.65, 0.05, μσ = true)
#     del     ~ MacroModelling.Beta(0.01, 0.005, μσ = true)
#     z_e_a   ~ MacroModelling.InverseGamma(0.035449, Inf, μσ = true)
#     z_e_m   ~ MacroModelling.InverseGamma(0.008862, Inf, μσ = true)
#     # println([alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])
#     Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])
# end
# FS2000_loglikelihood = FS2000_loglikelihood_function(data, FS2000, observables)
# n_samples = 1000
# chain_NUTS  = sample(FS2000_loglikelihood, NUTS(), n_samples, progress = false);

# Smets Wouters 2007

@model SW03 begin
    -q[0] + beta * ((1 - tau) * q[1] + epsilon_b[1] * (r_k[1] * z[1] - psi^-1 * r_k[ss] * (-1 + exp(psi * (-1 + z[1])))) * (C[1] - h * C[0])^(-sigma_c))
    
    -q_f[0] + beta * ((1 - tau) * q_f[1] + epsilon_b[1] * (r_k_f[1] * z_f[1] - psi^-1 * r_k_f[ss] * (-1 + exp(psi * (-1 + z_f[1])))) * (C_f[1] - h * C_f[0])^(-sigma_c))
    
    -r_k[0] + alpha * epsilon_a[0] * mc[0] * L[0]^(1 - alpha) * (K[-1] * z[0])^(-1 + alpha)
    
    -r_k_f[0] + alpha * epsilon_a[0] * mc_f[0] * L_f[0]^(1 - alpha) * (K_f[-1] * z_f[0])^(-1 + alpha)
    
    -G[0] + T[0]
    
    -G[0] + G_bar * epsilon_G[0]
    
    -G_f[0] + T_f[0]
    
    -G_f[0] + G_bar * epsilon_G[0]
    
    -L[0] + nu_w[0]^-1 * L_s[0]
    
    -L_s_f[0] + L_f[0] * (W_i_f[0] * W_f[0]^-1)^(lambda_w^-1 * (-1 - lambda_w))
    
    L_s_f[0] - L_f[0]
    
    L_s_f[0] + lambda_w^-1 * L_f[0] * W_f[0]^-1 * (-1 - lambda_w) * (-W_disutil_f[0] + W_i_f[0]) * (W_i_f[0] * W_f[0]^-1)^(-1 + lambda_w^-1 * (-1 - lambda_w))
    
    Pi_ws_f[0] - L_s_f[0] * (-W_disutil_f[0] + W_i_f[0])
    
    Pi_ps_f[0] - Y_f[0] * (-mc_f[0] + P_j_f[0]) * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p))
    
    -Q[0] + epsilon_b[0]^-1 * q[0] * (C[0] - h * C[-1])^(sigma_c)
    
    -Q_f[0] + epsilon_b[0]^-1 * q_f[0] * (C_f[0] - h * C_f[-1])^(sigma_c)
    
    -W[0] + epsilon_a[0] * mc[0] * (1 - alpha) * L[0]^(-alpha) * (K[-1] * z[0])^alpha
    
    -W_f[0] + epsilon_a[0] * mc_f[0] * (1 - alpha) * L_f[0]^(-alpha) * (K_f[-1] * z_f[0])^alpha
    
    -Y_f[0] + Y_s_f[0]
    
    Y_s[0] - nu_p[0] * Y[0]
    
    -Y_s_f[0] + Y_f[0] * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p))
    
    beta * epsilon_b[1] * (C_f[1] - h * C_f[0])^(-sigma_c) - epsilon_b[0] * R_f[0]^-1 * (C_f[0] - h * C_f[-1])^(-sigma_c)
    
    beta * epsilon_b[1] * pi[1]^-1 * (C[1] - h * C[0])^(-sigma_c) - epsilon_b[0] * R[0]^-1 * (C[0] - h * C[-1])^(-sigma_c)
    
    Y_f[0] * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p)) - lambda_p^-1 * Y_f[0] * (1 + lambda_p) * (-mc_f[0] + P_j_f[0]) * P_j_f[0]^(-1 - lambda_p^-1 * (1 + lambda_p))
    
    epsilon_b[0] * W_disutil_f[0] * (C_f[0] - h * C_f[-1])^(-sigma_c) - omega * epsilon_b[0] * epsilon_L[0] * L_s_f[0]^sigma_l
    
    -1 + xi_p * (pi[0]^-1 * pi[-1]^gamma_p)^(-lambda_p^-1) + (1 - xi_p) * pi_star[0]^(-lambda_p^-1)
    
    -1 + (1 - xi_w) * (w_star[0] * W[0]^-1)^(-lambda_w^-1) + xi_w * (W[-1] * W[0]^-1)^(-lambda_w^-1) * (pi[0]^-1 * pi[-1]^gamma_w)^(-lambda_w^-1)
    
    -Phi - Y_s[0] + epsilon_a[0] * L[0]^(1 - alpha) * (K[-1] * z[0])^alpha
    
    -Phi - Y_f[0] * P_j_f[0]^(-lambda_p^-1 * (1 + lambda_p)) + epsilon_a[0] * L_f[0]^(1 - alpha) * (K_f[-1] * z_f[0])^alpha
    
    std_eta_b * eta_b[x] - log(epsilon_b[0]) + rho_b * log(epsilon_b[-1])
    
    -std_eta_L * eta_L[x] - log(epsilon_L[0]) + rho_L * log(epsilon_L[-1])
    
    std_eta_I * eta_I[x] - log(epsilon_I[0]) + rho_I * log(epsilon_I[-1])
    
    std_eta_w * eta_w[x] - f_1[0] + f_2[0]
    
    std_eta_a * eta_a[x] - log(epsilon_a[0]) + rho_a * log(epsilon_a[-1])
    
    std_eta_p * eta_p[x] - g_1[0] + g_2[0] * (1 + lambda_p)
    
    std_eta_G * eta_G[x] - log(epsilon_G[0]) + rho_G * log(epsilon_G[-1])
    
    -f_1[0] + beta * xi_w * f_1[1] * (w_star[0]^-1 * w_star[1])^(lambda_w^-1) * (pi[1]^-1 * pi[0]^gamma_w)^(-lambda_w^-1) + epsilon_b[0] * w_star[0] * L[0] * (1 + lambda_w)^-1 * (C[0] - h * C[-1])^(-sigma_c) * (w_star[0] * W[0]^-1)^(-lambda_w^-1 * (1 + lambda_w))
    
    -f_2[0] + beta * xi_w * f_2[1] * (w_star[0]^-1 * w_star[1])^(lambda_w^-1 * (1 + lambda_w) * (1 + sigma_l)) * (pi[1]^-1 * pi[0]^gamma_w)^(-lambda_w^-1 * (1 + lambda_w) * (1 + sigma_l)) + omega * epsilon_b[0] * epsilon_L[0] * (L[0] * (w_star[0] * W[0]^-1)^(-lambda_w^-1 * (1 + lambda_w)))^(1 + sigma_l)
    
    -g_1[0] + beta * xi_p * pi_star[0] * g_1[1] * pi_star[1]^-1 * (pi[1]^-1 * pi[0]^gamma_p)^(-lambda_p^-1) + epsilon_b[0] * pi_star[0] * Y[0] * (C[0] - h * C[-1])^(-sigma_c)
    
    -g_2[0] + beta * xi_p * g_2[1] * (pi[1]^-1 * pi[0]^gamma_p)^(-lambda_p^-1 * (1 + lambda_p)) + epsilon_b[0] * mc[0] * Y[0] * (C[0] - h * C[-1])^(-sigma_c)
    
    -nu_w[0] + (1 - xi_w) * (w_star[0] * W[0]^-1)^(-lambda_w^-1 * (1 + lambda_w)) + xi_w * nu_w[-1] * (W[-1] * pi[0]^-1 * W[0]^-1 * pi[-1]^gamma_w)^(-lambda_w^-1 * (1 + lambda_w))
    
    -nu_p[0] + (1 - xi_p) * pi_star[0]^(-lambda_p^-1 * (1 + lambda_p)) + xi_p * nu_p[-1] * (pi[0]^-1 * pi[-1]^gamma_p)^(-lambda_p^-1 * (1 + lambda_p))
    
    -K[0] + K[-1] * (1 - tau) + I[0] * (1 - 0.5 * varphi * (-1 + I[-1]^-1 * epsilon_I[0] * I[0])^2)
    
    -K_f[0] + K_f[-1] * (1 - tau) + I_f[0] * (1 - 0.5 * varphi * (-1 + I_f[-1]^-1 * epsilon_I[0] * I_f[0])^2)
    
    U[0] - beta * U[1] - epsilon_b[0] * ((1 - sigma_c)^-1 * (C[0] - h * C[-1])^(1 - sigma_c) - omega * epsilon_L[0] * (1 + sigma_l)^-1 * L_s[0]^(1 + sigma_l))
    
    U_f[0] - beta * U_f[1] - epsilon_b[0] * ((1 - sigma_c)^-1 * (C_f[0] - h * C_f[-1])^(1 - sigma_c) - omega * epsilon_L[0] * (1 + sigma_l)^-1 * L_s_f[0]^(1 + sigma_l))
    
    -epsilon_b[0] * (C[0] - h * C[-1])^(-sigma_c) + q[0] * (1 - 0.5 * varphi * (-1 + I[-1]^-1 * epsilon_I[0] * I[0])^2 - varphi * I[-1]^-1 * epsilon_I[0] * I[0] * (-1 + I[-1]^-1 * epsilon_I[0] * I[0])) + beta * varphi * I[0]^-2 * epsilon_I[1] * q[1] * I[1]^2 * (-1 + I[0]^-1 * epsilon_I[1] * I[1])
    
    -epsilon_b[0] * (C_f[0] - h * C_f[-1])^(-sigma_c) + q_f[0] * (1 - 0.5 * varphi * (-1 + I_f[-1]^-1 * epsilon_I[0] * I_f[0])^2 - varphi * I_f[-1]^-1 * epsilon_I[0] * I_f[0] * (-1 + I_f[-1]^-1 * epsilon_I[0] * I_f[0])) + beta * varphi * I_f[0]^-2 * epsilon_I[1] * q_f[1] * I_f[1]^2 * (-1 + I_f[0]^-1 * epsilon_I[1] * I_f[1])

    -C[0] - I[0] - T[0] + Y[0] - psi^-1 * r_k[ss] * K[-1] * (-1 + exp(psi * (-1 + z[0])))

    -C_f[0] - I_f[0] + Pi_ws_f[0] - T_f[0] + Y_f[0] + L_s_f[0] * W_disutil_f[0] - L_f[0] * W_f[0] - psi^-1 * r_k_f[ss] * K_f[-1] * (-1 + exp(psi * (-1 + z_f[0])))
    
    epsilon_b[0] * (K[-1] * r_k[0] - r_k[ss] * K[-1] * exp(psi * (-1 + z[0]))) * (C[0] - h * C[-1])^(-sigma_c)
    
    epsilon_b[0] * (K_f[-1] * r_k_f[0] - r_k_f[ss] * K_f[-1] * exp(psi * (-1 + z_f[0]))) * (C_f[0] - h * C_f[-1])^(-sigma_c)


    # Perceived inflation objective
    std_eta_pi * eta_pi[x] - log(pi_obj[0]) + rho_pi_bar * log(pi_obj[-1]) + log(calibr_pi_obj) * (1 - rho_pi_bar)

    # Taylor rule
    -calibr_pi + std_eta_R * eta_R[x] - log(R[ss]^-1 * R[0]) + r_Delta_pi * (-log(pi[ss]^-1 * pi[-1]) + log(pi[ss]^-1 * pi[0])) + r_Delta_y * (-log(Y[ss]^-1 * Y[-1]) + log(Y[ss]^-1 * Y[0]) + log(Y_f[ss]^-1 * Y_f[-1]) - log(Y_f[ss]^-1 * Y_f[0])) + rho * log(R[ss]^-1 * R[-1]) + (1 - rho) * (log(pi_obj[0]) + r_pi * (-log(pi_obj[0]) + log(pi[ss]^-1 * pi[-1])) + r_Y * (log(Y[ss]^-1 * Y[0]) - log(Y_f[ss]^-1 * Y_f[0])))

	# Some observation equations
    R_obs[0]  = log(R[0])
    Y_obs[0]  = log(Y[0]/Y[-1])
    C_obs[0]  = log(C[0]/C[-1])
    I_obs[0]  = log(I[0]/I[-1])
	pi_obs[0] = log(pi[0])
    W_obs[0]  = log(W[0]/W[-1])
    L_obs[0]  = log(L[0]/L[-1])
	
    Rʸ_bps[0] = log(R[0]) * 40000
    πʸ_bps[0] = (log(pi[0]) + log(pi[-1]) + log(pi[-2]) + log(pi[-3])) * 10000
end


@parameters SW03 begin  
    lambda_p = .368
    G_bar = .362
    lambda_w = 0.5
    Phi = .819

    alpha = 0.3
    beta = 0.99
    gamma_w = 0.763
    gamma_p = 0.469
    h = 0.573
    omega = 1
    psi = 0.169

    r_pi = 1.684
    r_Y = 0.099
    r_Delta_pi = 0.14
    r_Delta_y = 0.159

    sigma_c = 1.353
    sigma_l = 2.4
    tau = 0.025
    varphi = 6.771
    xi_w = 0.737
    xi_p = 0.908

    rho = 0.961
    rho_b = 0.855
    rho_L = 0.889
    rho_I = 0.927
    rho_a = 0.823
    rho_G = 0.949
    rho_pi_bar = 0.924

    std_scaling_factor = 1

    std_eta_b = 0.336 / std_scaling_factor
    std_eta_L = 3.52 / std_scaling_factor
    std_eta_I = 0.085 / std_scaling_factor
    std_eta_a = 0.598 / std_scaling_factor
    std_eta_w = 0.6853261 / std_scaling_factor
    std_eta_p = 0.7896512 / std_scaling_factor
    std_eta_G = 0.325 / std_scaling_factor
    std_eta_R = 0.081 / std_scaling_factor
    std_eta_pi = 0.017 / std_scaling_factor
	
	pī = 1.005
	
    calibr_pi_obj | pī = pi_obj[ss]
    calibr_pi | pi[ss] = pi_obj[ss]
end
dt = CSV.read("C:/Users/fm007/Documents/GitHub/MacroModelling.jl/Workshop/EA_SW_data_growth.csv", DataFrame)

nms = names(dt)[[2:8...,11]]
transform!(dt, [:gdp_rpc, :conso_rpc, :inves_rpc, :wage_rph, :hours_pc, :employ] .=> (col -> col .- mean(col)) .=> [:gdp_rpc, :conso_rpc, :inves_rpc, :wage_rph, :hours_pc, :employ])
data = KeyedArray(Array(dt[:,2:end])',Variable = [:R_obs,:Y_obs,:C_obs,:I_obs,:pi_obs,:W_obs,:hours,:inv_defl,:cons_defl,:L_obs],Time = 1:size(dt)[1])
subdata = data(sort([:R_obs,:Y_obs,:C_obs,:I_obs,:pi_obs,:W_obs,:L_obs]),:)
filtered_shocksSW = get_estimated_shocks(SW03, subdata)


StatsPlots.plot(Distributions.Normal(0,1), fill=(0, .5,:blue), label= "Theoretical - Standard Normal")
        StatsPlots.density!(filtered_shocksSW[4,:], label = "Empirical - Filtered shocks distribution",title = title = "Monetary policy shock distributions - SW03")
