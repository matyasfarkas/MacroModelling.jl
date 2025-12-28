/*
 * This file replicates the estimation of the cash in advance model (termed M1
 * in the paper) described in Frank Schorfheide (2000): "Loss function-based
 * evaluation of DSGE models", Journal of Applied Econometrics, 15(6), 645-670.
 *
 * The data are taken from the replication package at
 * http://dx.doi.org/10.15456/jae.2022314.0708799949
 *
 * The prior distribution follows the one originally specified in Schorfheide's
 * paper. Note that the elicited beta prior for rho in the paper
 * implies an asymptote and corresponding prior mode at 0. It is generally
 * recommended to avoid this extreme type of prior.
 *
 * Because the data are already logged and we use the loglinear option to conduct
 * a full log-linearization, we need to use the logdata option.
 *
 * The equations are taken from J. Nason and T. Cogley (1994): "Testing the
 * implications of long-run neutrality for monetary business cycle models",
 * Journal of Applied Econometrics, 9, S37-S70, NC in the following.
 * Note that there is an initial minus sign missing in equation (A1), p. S63.
 *
 * This implementation was originally written by Michel Juillard. Please note that the
 * following copyright notice only applies to this Dynare implementation of the
 * model.
 */

/*
 * Copyright © 2004-2023 Dynare Team
 *
 * This file is part of Dynare.
 *
 * Dynare is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Dynare is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Dynare.  If not, see <https://www.gnu.org/licenses/>.
 */

var m       ${m}$           (long_name='money growth')
    P       ${P}$           (long_name='Price level')
    c       ${c}$           (long_name='consumption')
    e       ${e}$           (long_name='capital stock')
    W       ${W}$           (long_name='Wage rate')
    R       ${R}$           (long_name='interest rate')
    k       ${k}$           (long_name='capital stock')
    d       ${d}$           (long_name='dividends')
    n       ${n}$           (long_name='labor')
    l       ${l}$           (long_name='loans')
    gy_obs  ${\Delta \ln GDP}$  (long_name='detrended capital stock')
    gp_obs  ${\Delta \ln P}$    (long_name='detrended capital stock')
    y       ${y}$           (long_name='detrended output')
    dA      ${\Delta A}$    (long_name='TFP growth')
    ;
varexo e_a  ${\epsilon_A}$      (long_name='TFP shock')
    e_m     ${\epsilon_M}$      (long_name='Money growth shock')
    ;

parameters alp  ${\alpha}$       (long_name='capital share')
    bet         ${\beta}$        (long_name='discount factor')
    gam         ${\gamma}$       (long_name='long-run TFP growth')
    logmst         ${\log(m^*)}$ (long_name='long-run money growth')
    rho         ${\rho}$         (long_name='autocorrelation money growth')
    phi         ${\phi}$         (long_name='labor weight in consumption')
    del         ${\delta}$       (long_name='depreciation rate')
    ;

% Table 1 posterior mode from Schorfheide (2000) - matching MacroModelling.jl
alp = 0.356;
bet = 0.993;
gam = 0.0085;
logmst = log(1.0002);
rho = 0.129;
phi = 0.65;
del = 0.01;

model;
[name='NC before eq. (1), TFP growth equation']
dA = exp(gam+e_a);
[name='NC eq. (2), money growth rate']
log(m) = (1-rho)*logmst + rho*log(m(-1))+e_m;
[name='NC eq. (A1), Euler equation']
-P/(c(+1)*P(+1)*m)+bet*P(+1)*(alp*exp(-alp*(gam+log(e(+1))))*k^(alp-1)*n(+1)^(1-alp)+(1-del)*exp(-(gam+log(e(+1)))))/(c(+2)*P(+2)*m(+1))=0;
[name='NC below eq. (A1), firm borrowing constraint']
W = l/n;
[name='NC eq. (A2), intratemporal labour market condition']
-(phi/(1-phi))*(c*P/(1-n))+l/n = 0;
[name='NC below eq. (A2), credit market clearing']
R = P*(1-alp)*exp(-alp*(gam+e_a))*k(-1)^alp*n^(-alp)/W;
[name='NC eq. (A3), credit market optimality']
1/(c*P)-bet*P*(1-alp)*exp(-alp*(gam+e_a))*k(-1)^alp*n^(1-alp)/(m*l*c(+1)*P(+1)) = 0;
[name='NC eq. (18), aggregate resource constraint']
c+k = exp(-alp*(gam+e_a))*k(-1)^alp*n^(1-alp)+(1-del)*exp(-(gam+e_a))*k(-1);
[name='NC eq. (19), money market condition']
P*c = m;
[name='NC eq. (20), credit market equilibrium condition']
m-1+d = l;
[name='Definition TFP shock']
e = exp(e_a);
[name='Implied by NC eq. (18), production function']
y = k(-1)^alp*n^(1-alp)*exp(-alp*(gam+e_a));
[name='Observation equation GDP growth']
gy_obs = dA*y/y(-1);
[name='Observation equation price level']
gp_obs = (P/P(-1))*m(-1)/dA;
end;

shocks;
var e_a; stderr 0.035449;
var e_m; stderr 0.008862;
end;

steady_state_model;
  dA = exp(gam);
  gst = 1/dA;
  m = exp(logmst);
  khst = ( (1-gst*bet*(1-del)) / (alp*gst^alp*bet) )^(1/(alp-1));
  xist = ( ((khst*gst)^alp - (1-gst*(1-del))*khst)/m )^(-1);
  nust = phi*m^2/( (1-alp)*(1-phi)*bet*gst^alp*khst^alp );
  n  = xist/(nust+xist);
  P  = xist + nust;
  k  = khst*n;

  l  = phi*m*n/( (1-phi)*(1-n) );
  c  = m/P;
  d  = l - m + 1;
  y  = k^alp*n^(1-alp)*gst^alp;
  R  = m/bet;
  W  = l/n;
  ist  = y-c;
  q  = 1 - d;

  e = 1;

  gp_obs = m/dA;
  gy_obs = dA;
end;


steady;
check;

% Extended Path (SEP) tests for validation against MacroModelling.jl
% Configure extended_path options for better convergence
options_.ep.stochastic.order = 0;
options_.ep.verbosity = 1;
options_.ep.maxit = 500;
options_.ep.tolerance.f = 1e-5;
options_.ep.tolerance.x = 1e-5;
% Use trust region solver for better convergence
options_.solve_algo = 4;

% Test 1: SEP with order=1 (periods=10, branching length = 1)
disp('================================================================');
disp('DYNARE EXTENDED PATH TEST 1: periods=10, order=1');
disp('================================================================');
extended_path(periods=10, order=1);

% Report key steady state values for comparison
disp(' ');
disp('Key steady state values (SEP order=1):');
Y_idx = strmatch('y', M_.endo_names, 'exact');
C_idx = strmatch('c', M_.endo_names, 'exact');
R_idx = strmatch('R', M_.endo_names, 'exact');

if size(oo_.endo_simul, 2) >= M_.maximum_lag+1
    disp(['  y  = ' num2str(oo_.endo_simul(Y_idx, M_.maximum_lag+1), '%12.8f')]);
    disp(['  c  = ' num2str(oo_.endo_simul(C_idx, M_.maximum_lag+1), '%12.8f')]);
    disp(['  R  = ' num2str(oo_.endo_simul(R_idx, M_.maximum_lag+1), '%12.8f')]);
else
    disp('  Warning: Simulation did not complete successfully');
end

% Test 2: SEP with order=2 (periods=10, branching length = 2)
disp(' ');
disp('================================================================');
disp('DYNARE EXTENDED PATH TEST 2: periods=10, order=2');
disp('================================================================');
extended_path(periods=10, order=2);

% Report key steady state values for comparison
% Report key steady state values for comparison
disp(' ');
disp('Key steady state values (SEP order=1):');
Y_idx = strmatch('y', M_.endo_names, 'exact');
C_idx = strmatch('c', M_.endo_names, 'exact');
R_idx = strmatch('R', M_.endo_names, 'exact');

if size(oo_.endo_simul, 2) >= M_.maximum_lag+1
    disp(['  y  = ' num2str(oo_.endo_simul(Y_idx, M_.maximum_lag+1), '%12.8f')]);
    disp(['  c  = ' num2str(oo_.endo_simul(C_idx, M_.maximum_lag+1), '%12.8f')]);
    disp(['  R  = ' num2str(oo_.endo_simul(R_idx, M_.maximum_lag+1), '%12.8f')]);
else
    disp('  Warning: Simulation did not complete successfully');
end


disp(' ');
disp('================================================================');
disp('EXTENDED PATH TESTS COMPLETE');
disp('================================================================');

% ============================================================================
% IRF COMPUTATION: Conditional Forecast Method
% ============================================================================
% Compute IRFs as difference between shocked and baseline simulations
% Method: shocked(1σ at t=1, uncertain future) - baseline(0 at t=1, uncertain future)

disp(' ');
disp('================================================================');
disp('COMPUTING STOCHASTIC IRFs (Conditional Forecast Method)');
disp('================================================================');

% IRF horizon
irf_horizon = 20;

% Get shock standard deviations
sigma_e_a = 0.035449;
sigma_e_m = 0.008862;

% Get variable indices for reporting
Y_idx = strmatch('y', M_.endo_names, 'exact');
C_idx = strmatch('c', M_.endo_names, 'exact');
R_idx = strmatch('R', M_.endo_names, 'exact');
n_idx = strmatch('n', M_.endo_names, 'exact');
k_idx = strmatch('k', M_.endo_names, 'exact');

% ============================================================================
% IRF 1: TFP Shock (e_a)
% ============================================================================
disp(' ');
disp('Computing IRF to TFP shock (e_a, 1 sigma)...');

% BASELINE: Zero shock at t=1
perfect_foresight_setup(periods=irf_horizon);
% After setup, oo_.exo_simul has size (periods+2) x nshocks
% Set all shocks to zero
oo_.exo_simul(:,:) = 0;
perfect_foresight_solver;
baseline_e_a = oo_.endo_simul;

% SHOCKED: 1-sigma TFP shock at t=1
perfect_foresight_setup(periods=irf_horizon);
% Set all shocks to zero, then add shock at period 2 (period 1 is initial condition)
oo_.exo_simul(:,:) = 0;
oo_.exo_simul(2, 1) = sigma_e_a;  % e_a shock at t=1 (row 2 in exo_simul)
perfect_foresight_solver;
shocked_e_a = oo_.endo_simul;

% Compute IRF = difference
irf_e_a = shocked_e_a - baseline_e_a;

% Report IRF for key variables
disp('IRF to TFP shock (e_a, 1 sigma):');
disp('  Period    y           c           R           n           k');
disp('  --------------------------------------------------------------');
for t = 1:min(10, irf_horizon)
    fprintf('  %2d     %10.6f  %10.6f  %10.6f  %10.6f  %10.6f\n', ...
            t, ...
            irf_e_a(Y_idx, M_.maximum_lag+t), ...
            irf_e_a(C_idx, M_.maximum_lag+t), ...
            irf_e_a(R_idx, M_.maximum_lag+t), ...
            irf_e_a(n_idx, M_.maximum_lag+t), ...
            irf_e_a(k_idx, M_.maximum_lag+t));
end

% ============================================================================
% IRF 2: Money Growth Shock (e_m)
% ============================================================================
disp(' ');
disp('Computing IRF to money growth shock (e_m, 1 sigma)...');

% BASELINE: Already computed above (reuse baseline_e_a)

% SHOCKED: 1-sigma money growth shock at t=1
perfect_foresight_setup(periods=irf_horizon);
% Set all shocks to zero, then add shock at period 2 (period 1 is initial condition)
oo_.exo_simul(:,:) = 0;
oo_.exo_simul(2, 2) = sigma_e_m;  % e_m shock at t=1 (row 2, column 2 for e_m)
perfect_foresight_solver;
shocked_e_m = oo_.endo_simul;

% Compute IRF = difference
irf_e_m = shocked_e_m - baseline_e_a;

% Report IRF for key variables
disp('IRF to money growth shock (e_m, 1 sigma):');
disp('  Period    y           c           R           n           k');
disp('  --------------------------------------------------------------');
for t = 1:min(10, irf_horizon)
    fprintf('  %2d     %10.6f  %10.6f  %10.6f  %10.6f  %10.6f\n', ...
            t, ...
            irf_e_m(Y_idx, M_.maximum_lag+t), ...
            irf_e_m(C_idx, M_.maximum_lag+t), ...
            irf_e_m(R_idx, M_.maximum_lag+t), ...
            irf_e_m(n_idx, M_.maximum_lag+t), ...
            irf_e_m(k_idx, M_.maximum_lag+t));
end

disp(' ');
disp('================================================================');
disp('IRF COMPUTATION COMPLETE');
disp('================================================================');


//===============================================
// Estimation section
//===============================================
/*
% Table 1 of Schorfheide (2000)
estimated_params;
alp, beta_pdf, 0.356, 0.02;
bet, beta_pdf, 0.993, 0.002;
gam, normal_pdf, 0.0085, 0.003;
logmst, normal_pdf, 0.0002, 0.007;
rho, beta_pdf, 0.129, 0.223;
phi, beta_pdf, 0.65, 0.05;
del, beta_pdf, 0.01, 0.005;
stderr e_a, inv_gamma_pdf, 0.035449, inf;
stderr e_m, inv_gamma_pdf, 0.008862, inf;
end;

varobs gp_obs gy_obs;

estimation(order=1, datafile=fs2000_data, loglinear,logdata, mode_compute=4, mh_replic=20000, nodiagnostic, mh_nblocks=2, mh_jscale=0.8, mode_check);

%uncomment the following lines to generate LaTeX-code of the model equations
%write_latex_original_model(write_equation_tags);
%collect_latex_files; -->
*/
//==========================================================================