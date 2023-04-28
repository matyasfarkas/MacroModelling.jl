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
    Y_GAP_PCT;      // [8. ] Output gap in percent

varexo EPS_Y_GAP;   // Shock to the IS equation

//---------------------------------------------------------------------
// 2. Parameter declaration and calibration
//---------------------------------------------------------------------
//    NOTE: NK denotes parameters defined in the NK block, 
//          FV denotes parameters defined in the Financial Vulnerability block
//          C  denotes composite parameters which are functions of the NK parameters
//---------------------------------------------------------------------

//         1 NK   2 NK  3 NK     4 FV      5 C     6 C     7 FV       8 FV            9 C   10 NK  11 NK   12 NK  13 NK  14 FV    15 NK   16 FV      17 FV   
parameters alpha  beta  epsilon  gamma_eta kappa   lambda  lambda_eta lambda_eta_eta  omega phi    phi_pi  phi_y  sigma  sigma_y  theta   theta_eta  theta_y;

// New Keynesian Parameters (Gali (2015), Chapter 3; Table 1 in AEA P&P)
alpha           = 1/3;
beta            = 0.99;
epsilon         = 6;
phi             = 1;
phi_pi          = 1.5;
phi_y           = 0.125;
sigma           = 1;
theta           = 2/3;

// EA calibration for parameters governing the 
// Vulnerability and financial accelerator parameter values 

gamma_eta       = -534.96;
lambda_eta      =  -6.49;
lambda_eta_eta  = -18.55;
sigma_y         = 16.48;
theta_eta       = -4.38;
theta_y         = 14.46;

/* Original parameter for the US
Vulnerability and financial accelerator parameter values (Table 2 in AEA P&P)
gamma_eta       = 0.01;
lambda_eta      = 1.97;
lambda_eta_eta  =-1.01;
sigma_y         = 0.17;
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
model(linear); 

//[1]. Dynamic IS Curve
Y_GAP = Y_GAP(+1)-1/sigma*(I-PI(+1)) - gamma_eta*ETA - sigma_y*EPS_Y_GAP;

//[2]. Process for Financial Conditions
ETA = lambda_eta*ETA(-1) + lambda_eta_eta*ETA(-2) - theta_y*Y_GAP - theta_eta*Y_GAP(+1);
    
//[3]. New Keynesian Phillips Curve
PI = beta*PI(+1)+ kappa*Y_GAP;

//[4]. Interest Rate Rule
I = phi_pi*PI + phi_y*Y_GAP;

//---------------------------------------------------------------------
// Reporting variables
//---------------------------------------------------------------------
//[5]. First difference of percent output gap
D_Y_GAP_PCT  = (Y_GAP_PCT - Y_GAP_PCT(-1)); 

//[6]. OYA inflation rate (in percent)
PI_OYA  = (PI+PI(-1)+PI(-2)+PI(-3))*100;

//[7].  Output gap in percent
Y_GAP_PCT = Y_GAP*100;

//[8]. Change in OYA inflation (in percent)
D_PI_ANN_PCT = PI_OYA - PI_OYA(-1);
end;

steady_state_model;
D_PI_ANN_PCT   = 0;             // [1.] First difference of annualized percent inflation (either OYA or not)
D_Y_GAP_PCT    = 0;             // [2.] First difference of percent output gap
ETA            = 0;             // [3.] Financial conditions variable
I              = 0;             // [4.] Nominal interest rate (OQA, not annualized and not in percent)
PI             = 0;             // [5.] Inflation (OQA, not annualized and not in percent)
PI_OYA         = 0;             // [6.] Inflation (OYA, annualized and in percent)
Y_GAP          = 0;             // [7.] Output gap (not in percent)
Y_GAP_PCT      = 0;             // [8.] Output gap in percent
end;

//---------------------------------------------------------------------
// Note: these generate IRFs and replicate Figure 3.2 on p. 55
//---------------------------------------------------------------------
shocks;
var EPS_Y_GAP   = 1;    // Shock to the IS curve
end;

//This is used to generate the A and B matrices of the linear homoskedastic solution
stoch_simul(order = 1, irf=12);
