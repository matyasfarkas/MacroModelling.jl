function h = QMIPF_solve_ss(zeta,zeta_ST,s_gy,s_gy_ST,SS_Y,SS_C,chi,sigma,eta_0,K_N_ST,alpha,theta_p,chi_0_ST,SS_TAU_N_ST,SS_TAU_C_ST,varkappa_ST,SS_NU_ST,SS_N,theta_w,tau_p,tau_w)

%This version 8/21/2022

%Note: chi, sigma, eta_0, alpha, theta_p, theta_w, tau_p, tau_w are identical in home and foreign by construction
keyboard;
h = fzero(@(SS_N_ST) zeta/zeta_ST*((1-s_gy)*SS_Y-SS_C)*SS_N_ST^(chi*sigma) + (1-s_gy_ST+eta_0*s_gy_ST)*(K_N_ST)^alpha*SS_N_ST^(chi*sigma+1) -  ((1+tau_p)*(1+tau_w)*(1-alpha)/(1+theta_p)*(K_N_ST)^alpha*(1/chi_0_ST/(1+theta_w))*(1-SS_TAU_N_ST)/(1+SS_TAU_C_ST))^(sigma) / (1-varkappa_ST-SS_NU_ST)  , SS_N);
