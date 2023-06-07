// EA NKV model by Tobias Adrian, Fernando Duarte, Nellie Liang and Pawel Zabczyk (2020)
// This version was prepared by M. Farkas, ECB (29/12/2020) based on the original code for the US.

//---------------------------------------------------------------------
// 1. Variable declaration
//---------------------------------------------------------------------
//---------------------------------------------------------------------
//Note: model variables are in log deviations
//---------------------------------------------------------------------

var         
    k
    z
	k_obs
	%c
	%q;   
;


varexo EPSz;    //---------------------------------------------------------------------
// 2. Parameter declaration and calibration
//---------------------------------------------------------------------
//    NOTE: NK denotes parameters defined in the NK block, 
//          FV denotes parameters defined in the Financial Vulnerability block
//          C  denotes composite parameters which are functions of the NK parameters
//---------------------------------------------------------------------

parameters alpha  
			beta  
			sigma  
			rho 
			delta;
				

// New Keynesian Parameters (Gali (2015), Chapter 3; Table 1 in AEA P&P)
alpha           = 0.5;
beta            = 0.95;
sigma        	= 0.01;
rho             = 0.2;
delta         	= 0.02;


model; 
     % 1 / (- k + (1 - delta) * k(-1) +  exp(z(-1)) * k(-1)^alpha ) - (beta / (- k(+1) + (1 - delta) * k +  exp(z) * k^alpha )) * (alpha * exp(z) * k(-1)^(alpha - 1) + (1 - delta)) =0;
     % z = rho* z(-1) + sigma * EPSz;
	
	 %c = - k(+1)  + (1 - delta) * k + q;
     %q = exp(z) * k^alpha;
     (- k  + (1 - delta) * k(-1) + (exp(z(-1)) * k(-1)^alpha)) = ((- k(+1)  + (1 - delta) * k +(exp(z) * k^alpha))) / ((alpha * exp(z) * k ^(alpha - 1) + (1 - delta))*beta); %- (0.002650855157668)*EPSz ;
     
	 z =rho* z(-1) + sigma * EPSz;
	 k_obs = k;
end;

steady_state_model;
z              = 0;             
k =  (((1 / beta) - 1 + delta) / alpha)^(1 / (alpha - 1));
k_obs =  (((1 / beta) - 1 + delta) / alpha)^(1 / (alpha - 1));

%c =  (((1 / beta) - 1 + delta) / alpha)^(alpha / (alpha - 1)) -delta * (((1 / beta) - 1 + delta) / alpha)^(1 / (alpha - 1));
%q =  (((1 / beta) - 1 + delta) / alpha)^(alpha / (alpha - 1));

end;



shocks;

var EPSz  = 1;  end;

estimated_params;
// PARAM NAME, INITVAL, LB, UB, PRIOR_SHAPE, PRIOR_P1, PRIOR_P2, PRIOR_P3, PRIOR_P4, JSCALE
// PRIOR_SHAPE: BETA_PDF, GAMMA_PDF, NORMAL_PDF, INV_GAMMA_PDF
% monetary policy parameters
	/* 
	
	α = 0.25
    # β = 0.95
    # σ = 0.01
    # ρ = 0.2
    # δ = 0.02
	
	α ~ Turing.Uniform(0.15, 0.45)
    β ~ Turing.Uniform(0.92, 0.9999)
    δ ~ Turing.Uniform(0.0001, 0.1)
    σ ~ Turing.Uniform(0.0, 0.1)
    ρ ~ Turing.Uniform(0.0, 1.0)
    γ ~ Turing.Uniform(0.0, 1.5)
	
	*/
alpha,0.25,,,UNIFORM_PDF,0.15,0.45;
beta,0.95,,,UNIFORM_PDF,0.92,0.9999;
sigma,0.01,,,UNIFORM_PDF,0.0,0.1;
rho,0.2,,,UNIFORM_PDF,0.0,1.0;
delta,0.02,,,UNIFORM_PDF,0.0001,0.1;
end;


options_.TeX=1;  


varobs k_obs;
estimation(datafile=rbc_obs,mode_compute=0,mode_file = RBC_kz_mode,first_obs=1,nobs=20,mh_replic=100000,mh_nblocks=1,mh_jscale=0.3,mh_drop=0.1,smoother);
figure;
disp(' ');
disp('NOW I DO STABILITY MAPPING and prepare sample for Reduced form Mapping');
disp(' ');
disp('Press ENTER to continue'); pause(5);

dynare_sensitivity(redform=1,nodisplay,Nsam=512); //create sample of reduced form coefficients
// NOTE: since namendo is empty by default, 
// this call does not perform the mapping of reduced form coefficient: just prepares the sample

disp(' ');
disp('ANALYSIS OF REDUCED FORM COEFFICIENTS');
disp(' ');
disp('Press ENTER to continue'); pause(5);

dynare_sensitivity(nodisplay, load_stab=1,  // load previously generated sample analysed for stability
redform=1,  // do the reduced form mapping
threshold_redform=[-1 0],  // filter reduced form coefficients (default=[])
namendo=(:),  // evaluate relationships for pie and R (namendo=(:) for all variables)
namexo=(:),     // evaluate relationships with exogenous e_R (use namexo=(:) for all shocks)
namlagendo=(:),   // evaluate relationships with lagged R (use namlagendo=(:) for all lagged endogenous)
stab=0, // don't repeat again the stability mapping
Nsam=512);






