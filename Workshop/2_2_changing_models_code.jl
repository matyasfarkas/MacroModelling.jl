
#=
# Part 2: Changing models
In this part of the workshop we are going to use the Smets and Wouters 2003 model of the euro area and do some modifications.

Let's start by loading the model and geting some standard output to familiarise ourselves with the model. For example the NSSS, IRFs, model simulations, theoretical standard deviations, or variance decompositions and explore the options of the various functions.
=#
using MacroModelling

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

end

@parameters SW03 begin  
    lambda_p = 0.368
    G_bar    = 0.362
    lambda_w = 0.5
    Phi      = 0.819

    alpha   = 0.3
    beta    = 0.99
    gamma_w = 0.763
    gamma_p = 0.469
    h       = 0.573
    omega   = 1
    psi     = 0.169

    r_pi        = 1.684
    r_Y         = 0.099
    r_Delta_pi  = 0.14
    r_Delta_y   = 0.159

    sigma_c = 1.353
    sigma_l = 2.4
    tau     = 0.025
    varphi  = 6.771
    xi_w    = 0.737
    xi_p    = 0.908

    rho         = 0.961
    rho_b       = 0.855
    rho_L       = 0.889
    rho_I       = 0.927
    rho_a       = 0.823
    rho_G       = 0.949
    rho_pi_bar  = 0.924

    std_scaling_factor = 10

    std_eta_b   = 0.336     / std_scaling_factor
    std_eta_L   = 3.52      / std_scaling_factor
    std_eta_I   = 0.085     / std_scaling_factor
    std_eta_a   = 0.598     / std_scaling_factor
    std_eta_w   = 0.6853261 / std_scaling_factor
    std_eta_p   = 0.7896512 / std_scaling_factor
    std_eta_G   = 0.325     / std_scaling_factor
    std_eta_R   = 0.081     / std_scaling_factor
    std_eta_pi  = 0.017     / std_scaling_factor

    calibr_pi_obj | 1 = pi_obj[ss]
    calibr_pi | pi[ss] = pi_obj[ss]
end




#= ## Implement a time-varying perceived inflation objective 
Currently the inflation objective is a standard AR(1) process and we want to include a response to lagged inflation: 

rho_pi * log(pi[-1] / pi[-5])

Find the perceived inflation objcetive in the model (2nd last equation) and modify it accordingly in the @model part and add new parameter values in the @parameters part:
=#

# You can start by modifying this code here:
@model SW03_obj begin
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
    log(pi_obj[0]) =  rho_pi_bar * log(pi_obj[-1]) + log(calibr_pi_obj) * (1 - rho_pi_bar) + std_eta_pi * eta_pi[x]


    # Taylor rule
    -calibr_pi + std_eta_R * eta_R[x] - log(R[ss]^-1 * R[0]) + r_Delta_pi * (-log(pi[ss]^-1 * pi[-1]) + log(pi[ss]^-1 * pi[0])) + r_Delta_y * (-log(Y[ss]^-1 * Y[-1]) + log(Y[ss]^-1 * Y[0]) + log(Y_f[ss]^-1 * Y_f[-1]) - log(Y_f[ss]^-1 * Y_f[0])) + rho * log(R[ss]^-1 * R[-1]) + (1 - rho) * (log(pi_obj[0]) + r_pi * (-log(pi_obj[0]) + log(pi[ss]^-1 * pi[-1])) + r_Y * (log(Y[ss]^-1 * Y[0]) - log(Y_f[ss]^-1 * Y_f[0])))

end


@parameters SW03_obj begin  

    lambda_p = 0.368
    G_bar    = 0.362
    lambda_w = 0.5
    Phi      = 0.819

    alpha   = 0.3
    beta    = 0.99
    gamma_w = 0.763
    gamma_p = 0.469
    h       = 0.573
    omega   = 1
    psi     = 0.169

    r_pi        = 1.684
    r_Y         = 0.099
    r_Delta_pi  = 0.14
    r_Delta_y   = 0.159

    sigma_c = 1.353
    sigma_l = 2.4
    tau     = 0.025
    varphi  = 6.771
    xi_w    = 0.737
    xi_p    = 0.908

    rho         = 0.961
    rho_b       = 0.855
    rho_L       = 0.889
    rho_I       = 0.927
    rho_a       = 0.823
    rho_G       = 0.949
    rho_pi_bar  = 0.924

    std_scaling_factor = 10

    std_eta_b   = 0.336     / std_scaling_factor
    std_eta_L   = 3.52      / std_scaling_factor
    std_eta_I   = 0.085     / std_scaling_factor
    std_eta_a   = 0.598     / std_scaling_factor
    std_eta_w   = 0.6853261 / std_scaling_factor
    std_eta_p   = 0.7896512 / std_scaling_factor
    std_eta_G   = 0.325     / std_scaling_factor
    std_eta_R   = 0.081     / std_scaling_factor
    std_eta_pi  = 0.017     / std_scaling_factor

    calibr_pi_obj | 1 = pi_obj[ss]
    calibr_pi | pi[ss] = pi_obj[ss]
end


#= Generate some output to understand what these changes do. For example the theoretical model standard deviations will tell us about the influence of this change. =#



#=
## Implement alternative Taylor rule
Let's change to the Taylor rule so that it resembles the one in the NAWM model: 

log(R[0] / R[ss]) = rho * log(R[-1] / R[ss])  + (1 - rho) * (log(pi_obj[0] / pi_obj[ss]) + r_pi * (log(pi[0] / pi[ss]) - log(pi_obj[0] / pi_obj[ss])) + r_Y * log(Y[0] / Y_f[0]))+ r_Delta_pi * log(pi[0] / pi[-1]) + r_Delta_y * (log(Y[0] / Y[-1]) - log(Y_f[0] / Y_f[-1])) + std_eta_R * eta_R[x] + calibr_pi

with the following parameterisation:
rho         = 0.924
r_pi        = 2.75
r_Y         = 0.0325
r_Delta_pi  = 0.0418
r_Delta_y   = 0.104
=#


@model SW03_TR begin
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
    log(pi_obj[0]) =  rho_pi_bar * log(pi_obj[-1]) + log(calibr_pi_obj) * (1 - rho_pi_bar) + rho_pi * log(pi[-1] / pi[-5]) + std_eta_pi * eta_pi[x]


    # Taylor rule
	log(R[0] / R[ss]) = rho * log(R[-1] / R[ss])  + (1 - rho) * (log(pi_obj[0] / pi_obj[ss]) + r_pi * (log(pi[0] / pi[ss]) - log(pi_obj[0] / pi_obj[ss])) + r_Y * log(Y[0] / Y_f[0]))+ r_Delta_pi * log(pi[0] / pi[-1]) + r_Delta_y * (log(Y[0] / Y[-1]) - log(Y_f[0] / Y_f[-1])) + std_eta_R * eta_R[x] + calibr_pi

#    log(R[0] / R[ss]) = rho * log(R[-1] / R[ss])  + (1 - rho) * (log(pi_obj[0] / pi_obj[ss]) + r_pi * (log(pi[0] / pi[ss]) - log(pi_obj[0] / pi_obj[ss])) + r_Y * log(Y[0] / Y_f[0]))+ r_Delta_pi * log(pi[0] / pi[-1]) + r_Delta_y * (log(Y[0] / Y[-1]) - log(Y_f[0] / Y_f[-1])) + std_eta_R * eta_R[x] + calibr_pi

end


@parameters SW03_TR begin  
	rho_pi 	 = 0.5
	
    lambda_p = 0.368
    G_bar    = 0.362
    lambda_w = 0.5
    Phi      = 0.819

    alpha   = 0.3
    beta    = 0.99
    gamma_w = 0.763
    gamma_p = 0.469
    h       = 0.573
    omega   = 1
    psi     = 0.169

    rho         = 0.924
    r_pi        = 2.75
    r_Y         = 0.0325
    r_Delta_pi  = 0.0418
    r_Delta_y   = 0.104

    sigma_c = 1.353
    sigma_l = 2.4
    tau     = 0.025
    varphi  = 6.771
    xi_w    = 0.737
    xi_p    = 0.908

    rho_b       = 0.855
    rho_L       = 0.889
    rho_I       = 0.927
    rho_a       = 0.823
    rho_G       = 0.949
    rho_pi_bar  = 0.924

    std_scaling_factor = 10

    std_eta_b   = 0.336     / std_scaling_factor
    std_eta_L   = 3.52      / std_scaling_factor
    std_eta_I   = 0.085     / std_scaling_factor
    std_eta_a   = 0.598     / std_scaling_factor
    std_eta_w   = 0.6853261 / std_scaling_factor
    std_eta_p   = 0.7896512 / std_scaling_factor
    std_eta_G   = 0.325     / std_scaling_factor
    std_eta_R   = 0.081     / std_scaling_factor
    std_eta_pi  = 0.017     / std_scaling_factor

    calibr_pi_obj | 1 = pi_obj[ss]
    calibr_pi | pi[ss] = pi_obj[ss]
end

get_std(SW03)
get_std(SW03_TR)

get_std(SW03)([:pi, :R, :Y])
get_std(SW03_TR, parameters = :rho_pi =>0)([:pi, :R, :Y])
stds = get_std(SW03_TR, parameter_derivatives = :rho_pi)

#= Let's check what this does to the model behaviour by looking at model moments, variance decompositions... =#


#= What happens to the model standard deviations if we switch off the feddback from past inflation? Let's compare plots as well: =#



#= Plot version with feedback: =#


#=
## Implement stochastic volatility
Let's go back to the textbook Gali new keyensian model but with slightly downscaled shocksizes:
=#


@model Gali_2015_chapter_3_nonlinear begin
	W_real[0] = C[0] ^ siggma * N[0] ^ varphi

	Q[0] = betta * (C[1] / C[0]) ^ (-siggma) * Z[1] / Z[0] / Pi[1]

	R[0] = 1 / Q[0]

	Y[0] = A[0] * (N[0] / S[0]) ^ (1 - alppha)

	R[0] = Pi[1] * realinterest[0]

	R[0] = 1 / betta * Pi[0] ^ phi_pi * (Y[0] / Y[ss]) ^ phi_y * exp(nu[0])

	C[0] = Y[0]

	log(A[0]) = rho_a * log(A[-1]) + σ_a * eps_a[x]

	log(Z[0]) = rho_z * log(Z[-1]) - σ_z * eps_z[x]

	nu[0] = rho_nu * nu[-1] + σ_nu * eps_nu[x]

	MC[0] = W_real[0] / (S[0] * Y[0] * (1 - alppha) / N[0])

	1 = theta * Pi[0] ^ (epsilon - 1) + (1 - theta) * Pi_star[0] ^ (1 - epsilon)

	S[0] = (1 - theta) * Pi_star[0] ^ (( - epsilon) / (1 - alppha)) + theta * Pi[0] ^ (epsilon / (1 - alppha)) * S[-1]

	Pi_star[0] ^ (1 + epsilon * alppha / (1 - alppha)) = epsilon * x_aux_1[0] / x_aux_2[0] * (1 - tau) / (epsilon - 1)

	x_aux_1[0] = MC[0] * Y[0] * Z[0] * C[0] ^ (-siggma) + betta * theta * Pi[1] ^ (epsilon + alppha * epsilon / (1 - alppha)) * x_aux_1[1]

	x_aux_2[0] = Y[0] * Z[0] * C[0] ^ (-siggma) + betta * theta * Pi[1] ^ (epsilon - 1) * x_aux_2[1]

	M_real[0] = Y[0] / R[0] ^ eta
end


@parameters Gali_2015_chapter_3_nonlinear begin
	σ_a = .01
	
	σ_z = .01
	
	σ_nu = .01
	
	siggma = 1

	varphi = 5

	phi_pi = 1.5

	phi_y = 0.125

	theta = 0.75

	rho_nu = 0.5

	rho_z = 0.5

	rho_a = 0.9

	betta = 0.99

	eta = 3.77

	alppha = 0.25

	epsilon = 9

	tau = 0

end


#= Let' look at some standard output first: =#



#= Since we will be implementing stochastic volatility, we need to solve the model with higher order perturbation methods in order to see the effects of it. Given the nonlinear solution method we can also look at the stochastic steady state. =#
get_SS(Gali_2015_chapter_3_nonlinear)

#= Note that in order to get the derivatives ww have to define the parameters for which we want to define them. Check the help (?) for details on how to do this.

Last but not least let's compare it to the NSSS.=#




#= Now let's make the monetary policy shock have stochastic volatility.

The factor in front of the shock innovation, σ_nu, becomes a variable:

    nu[0] = rho_nu * nu[-1] + σ_nu[0] * eps_nu[x]

and this variable is driven by an AR(1) shock process:

    log(σ_nu[0]) = (1 - rho_σ_nu) * log(σ̄_nu) + rho_σ_nu * log(σ_nu[-1]) + σ_σ_nu * σ_eps_nu[x]

Do the modifications in the model below: 
=#


@model Gali_2015_sv begin
	W_real[0] = C[0] ^ siggma * N[0] ^ varphi

	Q[0] = betta * (C[1] / C[0]) ^ (-siggma) * Z[1] / Z[0] / Pi[1]

	R[0] = 1 / Q[0]

	Y[0] = A[0] * (N[0] / S[0]) ^ (1 - alppha)

	R[0] = Pi[1] * realinterest[0]

	R[0] = 1 / betta * Pi[0] ^ phi_pi * (Y[0] / Y[ss]) ^ phi_y * exp(nu[0])

	C[0] = Y[0]

	log(A[0]) = rho_a * log(A[-1]) + σ_a * eps_a[x]

	log(Z[0]) = rho_z * log(Z[-1]) - σ_z * eps_z[x]

	nu[0] = rho_nu * nu[-1] + σ_nu[0] *eps_nu[x]

    log(σ_nu[0]) = (1 - rho_σ_nu) * log(σ̄_nu) + rho_σ_nu * log(σ_nu[-1]) + σ_σ_nu * σ_eps_nu[x]
	
	MC[0] = W_real[0] / (S[0] * Y[0] * (1 - alppha) / N[0])

	1 = theta * Pi[0] ^ (epsilon - 1) + (1 - theta) * Pi_star[0] ^ (1 - epsilon)

	S[0] = (1 - theta) * Pi_star[0] ^ (( - epsilon) / (1 - alppha)) + theta * Pi[0] ^ (epsilon / (1 - alppha)) * S[-1]

	Pi_star[0] ^ (1 + epsilon * alppha / (1 - alppha)) = epsilon * x_aux_1[0] / x_aux_2[0] * (1 - tau) / (epsilon - 1)

	x_aux_1[0] = MC[0] * Y[0] * Z[0] * C[0] ^ (-siggma) + betta * theta * Pi[1] ^ (epsilon + alppha * epsilon / (1 - alppha)) * x_aux_1[1]

	x_aux_2[0] = Y[0] * Z[0] * C[0] ^ (-siggma) + betta * theta * Pi[1] ^ (epsilon - 1) * x_aux_2[1]

	M_real[0] = Y[0] / R[0] ^ eta
end


@parameters Gali_2015_sv begin
	σ_a = .01
	
	σ_z = .01
	
	#σ_nu = .01
	
	siggma = 1

	varphi = 5

	phi_pi = 1.5

	phi_y = 0.125

	theta = 0.75

	rho_nu = 0.5

	rho_z = 0.5

	rho_a = 0.9

	betta = 0.99

	eta = 3.77

	alppha = 0.25

	epsilon = 9

	tau = 0

    rho_σ_nu = 0.95

    σ̄_nu = .01

    σ_σ_nu= 1

end


#= Let's compare impulse responses for first, second, and third order =#

get_SS(Gali_2015_chapter_3_nonlinear)
get_SS(Gali_2015_sv)


using StatsPlots
plot_irf(Gali_2015_sv)
plot_solution(Gali_2015_chapter_3_nonlinear, :A)

plot_solution(Gali_2015_chapter_3_nonlinear, :nu , algorithm = [:first_order, :second_order, :third_order])
#= Note how the response changes for labor and marginal cost at higher order.
Now let's look at the more interesting stochastic volatility shock at second and third order perturbation solution. Note that the effect of stochastic volalitity at the (stochastic) steady state only appears at third order. =#



#= Finally we can look at the solution mapping of past states to present variables for the different perturbation order: =#
