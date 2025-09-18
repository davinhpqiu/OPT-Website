Please run "demonXXXX.m" to solve different problems 

This package offers 2 solvers for 1-bit compressive sensing problems  
based on algorithms proposed in the following 2 papers: 

GPSP-----------------------------------------------------------------------
   Shenglong Zhou, 
   Sparse SVM for sufficient data reduction, 
   IEEE Transactions on Pattern Analysis and Machine Intelligence, 44, 5560-5571, 2022.

NM01-----------------------------------------------------------------------
   Shenglong Zhou, Lili Pan, Naihua Xiu and Huoduo Qi, 
   Quadratic convergence of smoothing Newton's method for 0/1 loss optimization, 
   SIAM Journal on Optimization, 31, 3184–3211, 2021.

Please credit them if you use the code for your research.

===========================================================================
function out = OBCSpack(A,b,s,k,solver,pars)
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
%           pars.eps     - The parameter in the model        (default 1e-4)
%           pars.eta     - The penalty parameter       (default 0.01/ln(n))
%           pars.acc     - Acceleration is used if acc=1        (default 0)
%           pars.big     - Start with a bigger s if big=1       (default 1)
%           pars.maxit   - Maximum number of iterations       (default 1e3) 
%           pars.tol     - Tolerance of halting condition    (default 1e-8)
%           -------------  For NM01 solving (SFRO)-------------------------
%           pars.x0      - The initial point           (default zeros(n,1))
%           pars.q       - Parameter in the objective         (default 0.5)
%           pars.vareps  - Parameter in the objective         (default 0.5)
%           pars.epsilon - Parameter in the objective        (default 0.15)
%           pars.lam     - The penalty parameter                (default 1)
%           pars.tau     - A useful parameter                   (default 1) 
%           pars.maxit   - Maximum number of iterations       (default 1e3)  
% -------------------------------------------------------------------------
% Outputs:
%     out.sol:   The sparse solution x
%     out.time:  CPU time
%     out.iter:  Number of iterations
%     out.obj:   Objective function value at out.sol 
% -------------------------------------------------------------------------
% Send your comments and suggestions to <slzhou2021@163.com> 
% Warning: Accuracy may not be guaranteed !!!!! ! 
%--------------------------------------------------------------------------

% Below is one example that you can run
% =========================================================================
clc; close all; clear; addpath(genpath(pwd));

n     = 2000;          % Signal dimension 
m     = ceil(0.5*n);   % Number of measurements
s     = ceil(0.01*n);  % Sparsity level
nf    = 0.05;          % Noisy ratio
r     = 0.02;          % Flipping ratio
k     = ceil(r*m);

A     = randn(m,n);
T     = randperm(n,s);
xo    = zeros(n,1);                      
xo(T) = (1+rand(s,1)).*sign(randn(s,1));  
xo(T) = xo(T)/norm(xo(T));                 % True sparse solution
bo    = sign(A(:,T)*xo(T)+nf*randn(m,1));
h     = ones(m,1);                         % Flipping vector
T     = randperm(m,k); 
h(T)  = -h(T);
b     = bo.*h; 

solver = {'GPSP','NM01'};
out    = OBCSpack(A,b,s,k,solver{1});  
fprintf(' Time:                  %6.3f sec\n',out.time);
fprintf(' Absolue error:         %6.2f %%\n', norm(xo-out.sol)*100);
fprintf(' Signal-to-noise ratio: %6.2f\n',-10*log10(norm(xo-out.sol)^2));
fprintf(' Hamming distence:      %6.3f\n',nnz(sign(A*out.sol)-b)/m)
fprintf(' Hamming error:         %6.3f\n',nnz(sign(A*out.sol)-bo)/m)
PlotRecovery(xo,out.sol,[1000 450 500 250],1) 