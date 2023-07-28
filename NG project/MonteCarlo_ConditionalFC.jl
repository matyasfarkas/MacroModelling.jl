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

alp     = 0.356
bet     = 0.993
gam     = 0.0085
mst     = 1.0002
rho     = 0.129
psi     = 0.65
del     = 0.01
z_e_a   = 0.035449
z_e_m   = 0.008862
parameters= [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m]

using Random,AxisKeys , Distributions, StatsPlots

Random.seed!(1)
periods = 4
shockdist = Distributions.Normal(0,1) #  Turing.Beta(10,1) #
j = 0
simnumb = 100
mgrid =21
mmin = -2
mmax =2
mE4= zeros(simnumb*mgrid)
df =    DataFrame(m= zeros(simnumb*mgrid), Em4=ones(simnumb*mgrid)*-99999) 

for Ri in LinRange(mmin, mmax, mgrid)
    conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,18,2),Variables = FS2000.var, Periods =collect(1:2))
    conditions[16,2] = Ri
    #shockcond = KeyedArray(zeros(1,1),Variables = [:e_m], Periods = [1])    
    initial_cond= get_conditional_forecast(FS2000, conditions,conditions_in_levels = false,periods =1)  # shocks =     shockcond, 
    Random.seed!(1)
    
    for i = 1:simnumb
        j += 1
        shocksdraw = zeros(2,1)
        shocksdraw[:,1] = rand(shockdist,2,1)
        df[j,1] = Ri
        df[j,2] =  get_irf(FS2000,shocks = shocksdraw, parameters=parameters,algorithm=:third_order,variables = [:m], initial_state =collect(initial_cond[1:FS2000.timings.nVars,2]))[4]
    end

end

@df df violin((:m), :Em4, linewidth=1, side=:right,  label="Expected policy rate in 4 periods")



import RDatasets
singers = RDatasets.dataset("lattice", "singer")
@df df violin(string.(:m), :Em4, linewidth=0)
