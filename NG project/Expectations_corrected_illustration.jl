using MacroModelling
import Turing, StatsPlots , Plots, Random
import LinearAlgebra as ℒ


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


# draw from t scaled by approximate invariant variance) for the initial condition
m = FS2000

solution = get_solution(m, m.parameter_values, algorithm = :second_order)

calculate_covariance_ = MacroModelling.calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(1:m.timings.nVars))
long_run_covariance = calculate_covariance_(solution[2])

T =80
skew_param = 4

Random.seed!(12345) #Fix seed to reproduce data
shockdist = Turing.SkewNormal(0,1,skew_param) 

shocks = rand(Turing.Normal(0,1),1,T)
shocks =[ shocks;  rand(shockdist,1,T)] #  shocks = randn(1,periods)

simulated_data = get_irf(m,shocks = shocks, periods = 0, levels = true)#(:k,:,:) |>collect


# define loglikelihood model - KF
Turing.@model function loglikelihood_scaling_function_normal(m, data, observables, Ω)

    alp     = 0.356
     bet     = 0.993
     gam     = 0.0085
     #mst     = 1.0002
     rho     = 0.129
     psi     = 0.65
     del     = 0.01
     z_e_a   = 0.035449
     z_e_m   = 0.008862
    
    mst    ~ Turing.Normal(1,0.1) 
        algorithm = :second_order
        parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m]
    
        Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters)
    
       
end

Ω =  eps()
n_samples = 10000

loglikelihood_scaling = loglikelihood_scaling_function_normal(m, simulated_data(:,:,:Shock_matrix), [:log_gy_obs, :log_gp_obs], Ω) # Kalman
sampsn = Turing.sample(loglikelihood_scaling, Turing.NUTS(), n_samples, progress = true)#, init_params = sol

StatsPlots.plot(sampsn)

# Model with mean drift adjustment
@model FS2000_skewE begin
    dA[0] = exp(gam + z_e_a  *  e_a[x])

    log(m[0]) = (1 - rho) * log(mst)  +  rho * log(m[-1]) + z_e_m  *  (e_m[x] + skew_param/sqrt(skew_param^2+1)*sqrt(2/3.1415926535897))

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

@parameters FS2000_skewE begin
 alp     = 0.356
 bet     = 0.993
 gam     = 0.0085
 mst     = 1.0002
 rho     = 0.129
 psi     = 0.65
 del     = 0.01
 z_e_a   = 0.035449
 z_e_m   = 0.008862
 skew_param= 4
end

m1 = FS2000_skewE

# define loglikelihood model - KF
Turing.@model function loglikelihood_scaling_function(m, data, observables, Ω)

alp     = 0.356
 bet     = 0.993
 gam     = 0.0085
 #mst     = 1.0002
 rho     = 0.129
 psi     = 0.65
 del     = 0.01
 z_e_a   = 0.035449
 z_e_m   = 0.008862
 skew_param ~ Turing.Uniform(1,8) # skew_param= 4
 mst    ~ Turing.Normal(1,0.1) 
    algorithm = :first_order
    parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m, skew_param]

    Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters)

   
end

loglikelihood_scaling_skewcorr = loglikelihood_scaling_function(m1, simulated_data(:,:,:Shock_matrix), [:log_gy_obs, :log_gp_obs], Ω) # Kalman
samps = Turing.sample(loglikelihood_scaling_skewcorr, Turing.NUTS(), n_samples, progress = true)#, init_params = sol

StatsPlots.plot(samps)