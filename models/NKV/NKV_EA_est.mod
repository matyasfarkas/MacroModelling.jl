// EA NKV model by Tobias Adrian, Fernando Duarte, Nellie Liang and Pawel Zabczyk (2020)
// This version was prepared by M. Farkas, ECB (29/12/2020) based on the original code for the US.

//---------------------------------------------------------------------
// 1. Variable declaration
//---------------------------------------------------------------------
//---------------------------------------------------------------------
//Note: model variables are in log deviations
//---------------------------------------------------------------------

var D_PI_ANN_PCT    // [1. ] First difference of annualized percent inflation
    D_Y_GAP_PCT     // [2. ] First difference of percent output gap
    ETA             // [3. ] FCI variable
    I               // [4. ] Nominal interest rate (OQA, not annualized and not in percent)
    PI              // [5. ] Inflation (OQA, not annualized and not in percent)
    PI_OYA          // [6. ] Inflation OYA, in percent
    Y_GAP           // [7. ] Output gap
    Y_GAP_PCT       // [8. ] Output gap in percent
    VX              // [9. ] Vulnerability function
    ROBS;           // [10.] Nominal interest rate  

varexo EPS_Y_GAP    // Shock to the IS equation
       EPS_PI       // Shock to the PC 
       EPS_ETA      // Shock to the FV
       EPS_MP;       // Shock to the TR

//---------------------------------------------------------------------
// 2. Parameter declaration and calibration
//---------------------------------------------------------------------
//    NOTE: NK denotes parameters defined in the NK block, 
//          FV denotes parameters defined in the Financial Vulnerability block
//          C  denotes composite parameters which are functions of the NK parameters
//---------------------------------------------------------------------

//         1 NK   2 NK  3 NK     4 FV      5 C     6 C     7 FV       8 FV            9 C   10 NK  11 NK   12 NK  13 NK  14 FV    15 NK   16 FV      17 FV   
parameters alpha  beta  epsilon  gamma_eta kappa   lambda  lambda_eta lambda_eta_eta  omega phi    phi_pi  phi_y  sigma  sigma_y  theta   theta_eta  theta_y
sigma_eta
rho_eta1
rho_eta2
rho_eta3
zeta_y 
sigma_I
sigma_pc
Rbar
PibarSS
;

// New Keynesian Parameters (Gali (2015), Chapter 3; Table 1 in AEA P&P)
alpha           = 1/3;
beta            = 0.99;
epsilon         = 6;
phi             = 1;
phi_pi          = 1.5;
phi_y           = 0.125;
sigma           = 1;
theta           = 2/3;

// Vulnerability and financial accelerator parameter values (Table 2 in AEA P&P)
gamma_eta       = 0.01;
lambda_eta      = 1.97;
lambda_eta_eta  =-1.01;
sigma_y         = 0.17;
theta_eta       = 0.31;
theta_y         = 0.08;


// New shocks and VX parameters initialization
sigma_eta = 0.01;
sigma_I = 0.1;
sigma_pc = 0.1;

rho_eta1 = 0.75;
rho_eta2 = 0.25;
rho_eta3 = 0.25;
zeta_y = 0.75;
PibarSS           = 1.00475; 
Rbar              = (PibarSS-1)+(1.003/beta-1)+1;

/*
// EA calibration for parameters governing the 
// Vulnerability and financial accelerator parameter values 

gamma_eta       = -534.96;
lambda_eta      =  -6.49;
lambda_eta_eta  = -18.55;
//sigma_y         = 16.48;
theta_eta       = -4.38;
theta_y         = 14.46;
*/

/* Original parameter for the US
Vulnerability and financial accelerator parameter values (Table 2 in AEA P&P)
gamma_eta       = 0.01;
lambda_eta      = 1.97;
lambda_eta_eta  =-1.01;
// sigma_y         = 0.17;
theta_eta       = 0.31;
theta_y         = 0.08;

//Can use exact values from optimizer but it makes no difference to the results
//gamma_eta       = 0.010278228704065;
//lambda_eta      = 1.970913628521444;
//lambda_eta_eta  =-1.012021002994105;
//sigma_y         = 0.170377303938716;
//theta_eta       = 0.305787986569657;
//theta_y         = 0.075499203063294;

*/

// Composite parameters (Footnote 6 in AEA P&P)
omega           = (1-alpha)/(1-alpha+alpha*epsilon);
lambda          = (1-theta)*(1-beta*theta)/theta*omega;
kappa           = lambda*(sigma+(phi+alpha)/(1-alpha));

//---------------------------------------------------------------------
// 3. Model declaration
//---------------------------------------------------------------------
model; 

//[1]. Dynamic IS Curve
Y_GAP = Y_GAP(+1)-1/sigma*(I-PI(+1)) - gamma_eta*ETA - VX*EPS_Y_GAP;

//[2]. Process for Financial Conditions
ETA = lambda_eta*ETA(-1) + lambda_eta_eta*ETA(-2) - theta_y*Y_GAP - theta_eta*Y_GAP(+1) + sigma_eta * EPS_ETA ;

//[2a]
VX = sigma_y - rho_eta1  * ETA(-1) - rho_eta2  * ETA(-2) - rho_eta3  * ETA(-3) - zeta_y * Y_GAP(-1);

//[3]. New Keynesian Phillips Curve
PI = beta*PI(+1)+ kappa*Y_GAP + sigma_pc * EPS_PI; 

//[4]. Interest Rate Rule
I = phi_pi*PI + phi_y*Y_GAP + sigma_I * EPS_MP;



//---------------------------------------------------------------------
// Reporting variables
//---------------------------------------------------------------------
//[5]. First difference of percent output gap
D_Y_GAP_PCT  = (Y_GAP_PCT - Y_GAP_PCT(-1)); 

//[6]. OYA inflation rate (in percent)
PI_OYA  = (PibarSS-1)*100 +PI;
//= (PI+PI(-1)+PI(-2)+PI(-3))*100;

//[7].  Output gap in percent
Y_GAP_PCT = Y_GAP*100;

//[8]. Change in OYA inflation (in percent)
D_PI_ANN_PCT = PI_OYA - PI_OYA(-1);

//[9]. Observation equation interest rate (in percent)
ROBS =    4*(I) + (Rbar-1)*400;

end;

steady_state_model;
D_PI_ANN_PCT   = 0;             // [1.] First difference of annualized percent inflation (either OYA or not)
D_Y_GAP_PCT    = 0;             // [2.] First difference of percent output gap
ETA            = 0;             // [3.] Financial conditions variable
I              = 0;             // [4.] Nominal interest rate (OQA, not annualized and not in percent)
PI             = 0;             // [5.] Inflation (OQA, not annualized and not in percent)
PI_OYA         = (PibarSS-1)*100;             // [6.] Inflation (OYA, annualized and in percent)
Y_GAP          = 0;             // [7.] Output gap (not in percent)
Y_GAP_PCT      = 0;             // [8.] Output gap in percent
VX =  sigma_y;                  // [2a.] Vulnerability funciton SS
ROBS = (Rbar-1)*400;

end;

//---------------------------------------------------------------------
// Note: these generate IRFs and replicate Figure 3.2 on p. 55
//---------------------------------------------------------------------
shocks;
var EPS_Y_GAP   = 1;    // Shock to the IS curve
var EPS_PI      = 1;    // Shock to the PC 
var EPS_ETA     = 1;    // Shock to the FV
var  EPS_MP     = 1;    // Shock to the TR
end;

//This is used to generate the A and B matrices of the linear homoskedastic solution
// stoch_simul(order = 1, irf=12);


estimated_params;
// PARAM NAME, INITVAL, LB, UB, PRIOR_SHAPE, PRIOR_P1, PRIOR_P2, PRIOR_P3, PRIOR_P4, JSCALE
// PRIOR_SHAPE: BETA_PDF, GAMMA_PDF, NORMAL_PDF, INV_GAMMA_PDF


// Shock processes
sigma_y,		1,		    INV_GAMMA_PDF,	1,		0.7;
sigma_eta,		0.17,		INV_GAMMA_PDF,	0.17,		0.7;
sigma_pc,		0.17,		INV_GAMMA_PDF,	0.17,		0.7;
sigma_I,		0.17,		INV_GAMMA_PDF,	0.17,		0.7;

// Vulnerability function
rho_eta1, 0.75,		uniform_pdf, , , 0,1; //BETA_PDF,		0.75,		0.1;
rho_eta2, 0.25,		uniform_pdf, , , 0,1; //BETA_PDF,		0.25,		0.1;
rho_eta3, 0.25,		uniform_pdf, , , 0,1; //BETA_PDF,		0.25,		0.1;
zeta_y, 0.75,		uniform_pdf, , , 0,1; //BETA_PDF,		0.75,		0.1;

// New Keynesian Parameters (Gali (2015), Chapter 3; Table 1 in AEA P&P) are calibrated 

// Vulnerability and financial accelerator parameter values (Table 2 in AEA P&P) matched 
gamma_eta,      0.01, NORMAL_PDF, 0.01, 0.01;
lambda_eta,     1.97, NORMAL_PDF, 1.97, 1; 
lambda_eta_eta, -1.01, NORMAL_PDF, -1.01, 1; 
theta_eta,       0.31, NORMAL_PDF, 0.31, 1; 
theta_y,         0.08, NORMAL_PDF, 0.08, 1; 
end;

// For observed variable definitions please see the NAWM database's dataset construction files:
// NewDataConstFile_NAWM_dynare.m
// PI_OYA - is PIC qoq log growth rate of PCD -  100*(qdiff(log(nawm_dseries.PIC)))
// D_Y_GAP_PCT - is the output gap of the NAWM -  100 * (log(nawm_dseries_original.YER) - log(nawm_dseries.YGAP));
// ETA - is quarterly CISS
// ROBS - is just quarterly policy rate - STN from the projections database 


varobs PI_OYA D_Y_GAP_PCT ETA ROBS;

estimation(optim=('MaxIter',400),datafile=nkvea_data,mode_compute=6,first_obs=76, nobs = 97, presample=4,mh_replic=0,lik_init=2,prefilter=0);
//estimation(optim=('MaxIter',400),datafile=nkvea_data,mode_compute=0, mode_file = NKV_EA_est_mode,first_obs=76, nobs = 97, presample=4,mh_replic=10000,lik_init=2,prefilter=0);

stoch_simul(order=1, irf=20)  ROBS PI_OYA D_Y_GAP_PCT ETA;