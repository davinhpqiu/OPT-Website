function out = SCOpack(func,n,s,solvername,pars)
%--------------------------------------------------------------------------
% This code aims at solving the sparsity constrained optimization (SCO),
%
%         min_{x\in R^n} f(x),  s.t. ||x||_0<=s
%
% or non-negative and sparsity constrained optimization (NSCO):
%
%         min_{x\in R^n} f(x),  s.t. ||x||_0<=s, x>=0 
%
% where f: R^n->R and s<<n is an integer.
%--------------------------------------------------------------------------
% Inputs:
%   func:   A function handle defines                            (REQUIRED)
%                    (objective,gradient,sub-Hessain)
%   n:      Dimension of the solution x                          (REQUIRED)
%   s:      Sparsity level of x, an integer between 1 and n-1    (REQUIRED)
%   solver: A text string, can be one of {'NHTP','GPNP','IIHT'}  (REQUIRED)
%   pars  : ---------------For all solvers --------------------------------
%           pars.x0    --  Starting point of x         (default zeros(n,1))
%           pars.disp  --  =1 show results for each step        (default 1)
%                          =0 not show results for each step
%           pars.maxit --  Maximum number of iterations      (default  2e3) 
%           pars.tol   --  Tolerance of halting conditions   (default 1e-6)
%           pars.uppf  --  An upper bound of final objective (default -Inf)
%                          Useful for noisy case
%           ---------------Particular for NHTP ----------------------------
%           pars.eta   --  A positive scalar                    (default 1)  
%                          Tuning it may improve solution quality 
%           ---------------Particular for IIHT ----------------------------
%           pars.neg   --  =0 for model (SCO)                   (default 1)
%                          =1 for model (NSCO)
%--------------------------------------------------------------------------
% Outputs:
%     out.sol :   The sparse solution x
%     out.obj :   Objective function value at out.sol 
%     out.iter:   Number of iterations
%     out.time:   CPU time
%--------------------------------------------------------------------------
% Send your comments and suggestions to <<< slzhou2021@163.com >>>   
% WARNING: Accuracy may not be guaranteed!!!!!  
%--------------------------------------------------------------------------

warning off; 
if  nargin<4  
    disp(' Inputs are not enough !!! \n');
    return;
elseif nargin<5
    pars = []; 
end

solver = str2func(solvername);  
out    = solver(func,n,s,pars); 

end
