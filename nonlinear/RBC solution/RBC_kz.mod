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
	 
end;

steady_state_model;
z              = 0;             
k =  (((1 / beta) - 1 + delta) / alpha)^(1 / (alpha - 1));
%c =  (((1 / beta) - 1 + delta) / alpha)^(alpha / (alpha - 1)) -delta * (((1 / beta) - 1 + delta) / alpha)^(1 / (alpha - 1));
%q =  (((1 / beta) - 1 + delta) / alpha)^(alpha / (alpha - 1));

end;

shocks;

var EPSz  = 1;  end;
varobs z k;
dynare_sensitivity;

//This is used to generate the A and B matrices of the linear homoskedastic solution
stoch_simul(order = 1, irf=12);
