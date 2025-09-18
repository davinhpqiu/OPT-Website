function out = OBCSpack(A,b,s,k,solver,pars)
% -------------------------------------------------------------------------
% One-bit compressed sensing problem aims to recovere sparse signal x from
%
%                b = Diag(h).*sign( A*x + noise )
%
% 1) The double sparsity constrained optimization (DSCO)
%
%    min  ||Diag(b)*A*x+y-epsilon||^2 + eta||x||^2
%    s.t. ||x||_0<=s, ||y_+||_0<=k
%
% where (epsilon, eta)>0, s\in[1,n], k\in[0,m] are given.
%
% 2) The step function regularized optimization (SFRO)
%
%    min ||x.*x+vareps||^{q/2}_{q/2} + lam*||(epsilon - Diag(b)*A*x)_+||_0
%
% where (vareps, lam, epsilon)> 0, q\in(0,1).  
% -------------------------------------------------------------------------
% Inputs:
%  A:       The sensing matrix \in R^{m-by-n},                   (REQUIRED)
%  b:       The binary observation \in R^m, b_i\in{-1,1}         (REQUIRED)
%  s:       Sparsity level of x, an integer \in[1,n]             (REQUIRED)      
%  k:       An integer in [0,m], e.g., k = ceil(0.01m)           (REQUIRED)       
%  solver:  A text string, can be one of {'GPSP','NM01'}         (REQUIRED)            
%  pars:    Parameters are optional                              (OPTIONAL) 
%           ------------- For GPSP solving (DSCO)--------------------------
%           pars.eps     - The parameter in the model        (default,1e-4)
%           pars.eta     - The penalty parameter       (default,0.01/ln(n))
%           pars.acc     - Acceleration is used if acc=1        (default,0)
%           pars.big     - Start with a bigger s if big=1       (default,1)
%           pars.maxit   - Maximum number of iterations       (default,1e3) 
%           pars.tol     - Tolerance of halting condition    (default,1e-8)
%           -------------  For NM01 solving (SFRO)-------------------------
%           pars.x0      - The initial point           (default,zeros(n,1))
%           pars.q       - Parameter in the objective         (default,0.5)
%           pars.vareps  - Parameter in the objective         (default,0.5)
%           pars.epsilon - Parameter in the objective        (default,0.15)
%           pars.lam     - The penalty parameter                (default,1)
%           pars.tau     - A useful parameter                   (default,1) 
%           pars.maxit   - Maximum number of iterations       (default,1e3)  
% -------------------------------------------------------------------------
% Outputs:
%     out.sol:   The sparse solution x
%     out.time:  CPU time
%     out.iter:  Number of iterations
%     out.obj:   Objective function value at out.sol 
% -------------------------------------------------------------------------
% Send your comments and suggestions to <slzhou2021@163.com> 
% Warning: Accuracy may not be guaranteed !!!!! ! 
% -------------------------------------------------------------------------

warning off; 
if  nargin<5  
    disp(' Inputs are not enough !!! \n');
    return;
elseif nargin<6
    pars      = []; 
    pars.disp = 1;
end
 
if  isempty(s)
    solver = 'NM01';   
end

switch solver 
    case 'GPSP'; out = GPSP(A,b,s,k,pars);
    case 'NM01'; out = NM01(A,b,s,pars); 
end


end

