function function_output = QMIPF_SS(params)

%This version 8/21/2022

%This function takes parameters in the params vector and computes the model steady state
%which is then added to the params structure and exported as function_output

%First we need to unpack params so that we can refer to the parameters easily

%Manually create variables when Matlab same-named-functions exist
%Otherwise Matlab will try to run function instead. Even though eval doesn't seem to be used, probably related to 
%https:%www.mathworks.com/matlabcentral/answers/94543-why-does-my-code-not-work-correctly-if-i-use-the-eval-assignin-or-load-function-to-create-a-workspa
alpha = 0; beta  = 0; psi   = 0; sigma = 0;

%The parameters outside of the params vector exist in this function only and are not passed
import_from_params;

%We can now clear auxiliary variables
clearvars params;

% ----------------------------------------------------------------------------------------------------------------------|
%                                                  STEADY STATE DERIVATION                                              |
% ----------------------------------------------------------------------------------------------------------------------|
%This is the "contingent" part of the Steady State that may change when any of the underlying parameter values change

%----------------------------------------%
SS_GAMMA            = gamma_0*var_e^gamma_1;                                                   %S59.
k_n                 = ((1+theta_p)/(1+tau_p)*(1-beta)/(alpha*beta))^(1/(alpha-1));             % 91.
SS_MC_D             = (1+tau_p)/(1+theta_p);                                                   %S60.
SS_MC_D_ST          = (1+tau_p)/(1+theta_p);                                                   %S61.

%----------------------------------------%
k_n_ST              = ((1+theta_p)/(1+tau_p)*(1-beta_ST)/(alpha*beta_ST))^(1/(alpha-1));       % 92.  
b_y                 = (s_my - s_py) - 1/SS_GAMMA*(1-SS_TAU_F-beta/beta_ST); % 93.    
c_y                 = 1 - s_gy - b_y*(1-(1-omega_f)*(1-SS_TAU_F)*1/beta-omega_f*1/beta_ST) - ((1-SS_TAU_F)*1/beta-1/beta_ST)*((omega_f-omega_p)*s_py+(1-omega_f)*s_my);                                           %  94.
SS_N                = (1/((c_y + eta_0*s_gy)*(1-varkappa-SS_NU))*((1+tau_p)*(1+tau_w)*(1-alpha)/(1+theta_p)*(1/chi_0/(1+theta_w))*(1-SS_TAU_N)/(1+SS_TAU_C))^(sigma)*(k_n)^(alpha*(sigma-1)))^(1/(1+chi*sigma));  % S62.
k                   = k_n*SS_N;                                                                % 95.
SS_Y                = k_n^(alpha)*SS_N;                                                        %S63.
SS_C                = c_y*SS_Y;                                                                %S64.

%----------------------------------------%
%            Solve for SS N              %
%----------------------------------------%
SS_N_ST             = QMIPF_solve_SS(zeta,zeta_ST,s_gy,s_gy_ST,SS_Y,SS_C,chi,sigma,eta_0,k_n_ST,alpha,theta_p,chi_0_ST,SS_TAU_N_ST,SS_TAU_C_ST,varkappa_ST,SS_NU_ST,SS_N,theta_w,tau_p,tau_w); % S65.
                                  
k_ST                = k_n_ST*SS_N_ST;                                                                                                                                        % 96.
SS_Y_ST             = k_n_ST^(alpha)*SS_N_ST;                                                                                                                                % S66.
SS_C_ST             = zeta/zeta_ST*((1-s_gy)*SS_Y-SS_C) + (1-s_gy_ST)*k_n_ST^alpha*SS_N_ST;                                                                                  % S67.
SS_I                = SS_PI_C/beta;                                                                                                                                          % S68.
SS_I_ST             = SS_PI_C_ST/beta_ST;                                                                                                                                    % S69.
SS_IB               = SS_I;                                                                                                                                                  % S70.
SS_THETA            = 0;                                                                                                                                                     % S71.
SS_PI_D             = SS_PI_C;                                                                                                                                               % S72.
SS_PI_D_ST          = SS_PI_C_ST;                                                                                                                                            % S73.
SS_PI_M             = SS_PI_C;                                                                                                                                               % S74.
SS_PI_M_ST          = SS_PI_C_ST;                                                                                                                                            % S75.
SS_W_C              = (1+tau_p)/(1+theta_p)*(1-alpha)*k_n^(alpha);                                                                                                           % S76.
SS_W_C_ST           = (1+tau_p)/(1+theta_p)*(1-alpha)*k_n_ST^(alpha);                                                                                                        % S77.
SS_B                = b_y*SS_Y;                                                                                                                                              % S78.
SS_B_P              = s_py*SS_Y;                                                                                                                                             % S79.
SS_B_M              = s_my*SS_Y;                                                                                                                                             % S80.
SS_B_F              = -SS_B - SS_B_P + SS_B_M;                                                                                                                               % S81.
SS_G                = s_gy*SS_Y;                                                                                                                                             % S82.
SS_G_ST             = s_gy_ST*SS_Y_ST;                                                                                                                                       % S83.
SS_C_TIL            = SS_C+eta_0*SS_G;                                                                                                                                       % S84.
SS_C_TIL_ST         = SS_C_ST+eta_0*SS_G_ST;                                                                                                                                 % S85.
SS_LAM              = (SS_C_TIL-varkappa*SS_C_TIL-SS_C_TIL*SS_NU)^(-1/sigma)/(1+SS_TAU_C);                                                                                   % S86.
SS_LAM_ST           = (SS_C_TIL_ST-varkappa_ST*SS_C_TIL_ST-SS_C_TIL_ST*SS_NU_ST)^(-1/sigma)/(1+SS_TAU_C_ST);                                                                 % S87.
SS_W_TIL_C          = SS_W_C;                                                                                                                                                % S88.
SS_W_TIL_C_ST       = SS_W_C_ST;                                                                                                                                             % S89.
SS_PI_W             = SS_PI_C;                                                                                                                                               % S90.
SS_PI_W_ST          = SS_PI_C_ST;

%-----------------------------------------%                                                                                                                                
% Normalization factors for auxiliary variables from sticky wage block (needed for lmmcp)                                                                                   
z7_corr             = chi_0*SS_W_C^((1+theta_w)/theta_w*(1+chi))*SS_N^(1+chi);                                                                                               % 97.
z8_corr             = SS_LAM*(1-SS_TAU_N)*SS_W_C^((1+theta_w)/theta_w)*SS_N;                                                                                                 % 98.
z7_corr_ST          = chi_0_ST*SS_W_C_ST^((1+theta_w)/theta_w*(1+chi))*SS_N_ST^(1+chi);                                                                                      % 99.
z8_corr_ST          = SS_LAM_ST*(1-SS_TAU_N_ST)*SS_W_C_ST^((1+theta_w)/theta_w)*SS_N_ST;                                                                                     %100.
%-----------------------------------------%

SS_Z_7              = SS_VARSIGMA*chi_0*SS_W_C^((1+theta_w)/theta_w*(1+chi))*SS_N^(1+chi)/(1-beta*xi_w) /z7_corr;                                                            % S92.  
SS_Z_7_ST           = SS_VARSIGMA_ST*chi_0_ST*SS_W_C_ST^((1+theta_w)/theta_w*(1+chi))*SS_N_ST^(1+chi)/(1-beta*xi_w_ST) /z7_corr_ST;                                          % S93.  
SS_Z_8              = SS_VARSIGMA*SS_LAM*(1-SS_TAU_N)*(1+tau_w)*SS_W_C^((1+theta_w)/theta_w)*SS_N/(1-beta*xi_w) /z8_corr;                                                              % S94.  
SS_Z_8_ST           = SS_VARSIGMA_ST*SS_LAM_ST*(1-SS_TAU_N_ST)*(1+tau_w)*SS_W_C_ST^((1+theta_w)/theta_w)*SS_N_ST/(1-beta*xi_w_ST) /z8_corr_ST;                                         % S95.  
SS_PI_P             = SS_PI_D;                                                                                                                                               % S96.  
SS_PI_P_ST          = SS_PI_D_ST;                                                                                                                                            % S97.  
SS_PI_PM            = SS_PI_D;                                                                                                                                               % S98.  
SS_PI_PM_ST         = SS_PI_D_ST;                                                                                                                                            % S99.  
SS_U                = ( 1/(1-1/sigma)*(SS_C_TIL-varkappa*SS_C_TIL-SS_C_TIL*SS_NU)^(1-1/sigma) - chi_0*SS_W_AMP_U*SS_N^(1+chi)/(1+chi) )/(1-beta);                            %S100.
SS_U_ST             = ( 1/(1-1/sigma)*(SS_C_TIL_ST-varkappa_ST*SS_C_TIL_ST-SS_C_TIL_ST*SS_NU_ST)^(1-1/sigma) - chi_0_ST*SS_W_AMP_U_ST*SS_N_ST^(1+chi)/(1+chi) )/(1-beta_ST); %S101.
% Overwrite for log utility
if sigma==1
SS_U                = ( log(SS_C_TIL-varkappa*SS_C_TIL-SS_C_TIL*SS_NU) - chi_0*SS_W_AMP_U*SS_N^(1+chi)/(1+chi) )/(1-beta);                                                   %S100.
SS_U_ST             = ( log(SS_C_TIL_ST-varkappa_ST*SS_C_TIL_ST-SS_C_TIL_ST*SS_NU_ST) - chi_0_ST*SS_W_AMP_U_ST*SS_N_ST^(1+chi)/(1+chi) )/(1-beta_ST);                        %S101.
end

%-----------------------------------------%
% Here we set the home bias parameters for the foreign economy so that they are consistent with Q=1
s_omega_g_c_ST      = 0.1/0.29; %When adjusting we will want to preserve a reasonable ratio                                                                                 %101.
omega_c_ST          = zeta/zeta_ST* (SS_Y-(1-omega_c)*SS_C-(1-omega_g)*SS_G) / (SS_C_ST+s_omega_g_c_ST*SS_G_ST);                                                             %102.
omega_g_ST          = s_omega_g_c_ST*omega_c_ST;                                                                                                                             %103.

%-----------------------------------------%                                                                                                                                
SS_Y_M              = zeta/zeta_ST*(omega_c*SS_C+omega_g*SS_G);                                                                                                              %S102.
SS_Y_M_ST           = zeta_ST/zeta*(omega_c_ST*SS_C_ST+omega_g_ST*SS_G_ST);                                                                                                  %S103.
SS_Y_D              = SS_Y-SS_Y_M_ST;                                                                                                                                        %S104.
SS_Y_D_ST           = SS_Y_ST-SS_Y_M;                                                                                                                                        %S105.
SS_Z_1              = (1+psi)*(1+theta_p)/((1-beta*xi_p)*(1+psi+psi*theta_p))*SS_LAM*SS_Y_D*SS_MC_D;                                                               %S106.
SS_Z_1_ST           = (1+psi_ST)*(1+theta_p)/((1-beta_ST*xi_p_ST)*(1+psi_ST+psi_ST*theta_p))*SS_LAM_ST*SS_Y_D_ST*SS_MC_D_ST;                                       %S107.
SS_Z_2              = (1+tau_p)*SS_LAM*SS_Y_D/(1-beta*xi_p);                                                                                                                           %S108.
SS_Z_2_ST           = (1+tau_p)*SS_LAM_ST*SS_Y_D_ST/(1-beta_ST*xi_p_ST);                                                                                                               %S109.
SS_Z_3              = (1+tau_p)*(psi*theta_p)/(1+psi+psi*theta_p)*SS_LAM*SS_Y_D/(1-beta*xi_p);                                                                                         %S110.
SS_Z_3_ST           = (1+tau_p)*(psi_ST*theta_p)/(1+psi_ST+psi_ST*theta_p)*SS_LAM_ST*SS_Y_D_ST/(1-beta_ST*xi_p_ST);                                                                    %S111.
SS_Z_M_1            = (1+psi_m)*(1+theta_p)/((1-beta*xi_m)*(1+psi_m+psi_m*theta_p))*SS_LAM*SS_Y_M_ST*SS_MC_D;                                                      %S112.
SS_Z_M_1_ST         = (1+psi_m_ST)*(1+theta_p)/((1-beta_ST*xi_m_ST)*(1+psi_m_ST+psi_m_ST*theta_p))*SS_LAM_ST*SS_Y_M*SS_MC_D_ST;                                    %S113.
SS_Z_M_2            = SS_LAM*SS_Y_M_ST*(1+tau_p)/(1-beta*xi_m);                                                                                                                        %S114.
SS_Z_M_2_ST         = SS_LAM_ST*SS_Y_M*(1+tau_p)/(1-beta_ST*xi_m_ST);                                                                                                                  %S115.
SS_Z_M_3            = (psi_m*theta_p)/(1+psi_m+psi_m*theta_p)*SS_LAM*SS_Y_M_ST*(1+tau_p)/(1-beta*xi_m);                                                                                %S116.
SS_Z_M_3_ST         = (psi_m_ST*theta_p)/(1+psi_m_ST+psi_m_ST*theta_p)*SS_LAM_ST*SS_Y_M*(1+tau_p)/(1-beta_ST*xi_m_ST);                                                                 %S117.
SS_M_C              = omega_c*SS_C;                                                                                                                                          %S118.                                       
SS_M_C_ST           = omega_c_ST*SS_C_ST;                                                                                                                                    %S119.                                       
SS_M_G              = omega_g*SS_G;                                                                                                                                          %S120.                                       
SS_M_G_ST           = omega_g_ST*SS_G_ST;                                                                                                                                    %S121.                                       
                                                                                                                                            
m                   = -SS_B/SS_Y + 0.12*4;  % Debt limit 12 percent above steady state                                                                                      %104.                                        
SS_BLIM             = SS_B + m*SS_Y;

%Note: Number of SS variables is smaller than total number of model variables because some 
%      POT variables load on the values above

%After doing all the computations we can clear params as we no longer need it
clearvars params;

%Now export workspace contents to params vector
export_to_params;

function_output = params;

%Note: this will contain all the original parameters and all the newly computed steady state values
end
