Please run "demonXXXX.m" to solve different problems 

This package offers 1 solver for sparsity regularized optimization problems  
based on the algorithm proposed in the following paper: 

NL0R------------------------------------------------------------------------
    Shenglong Zhou, Lili Pan, and Naihua Xiu, 
    Newton method for l_0 regularized optimization,
    Numerical Algorithms, 88, 1541-1570, 2021

Please credit it if you use the code for your research.

===========================================================================
function out = NL0R(func,n,lambda,pars)
%--------------------------------------------------------------------------
% This code aims at solving the L0 norm regularized optimization 
%
%         min_{x\in R^n} f(x) + lambda*||x||_0^0
%
% where f: R^n->R, lambda>0
%       ||x||_0^0 counts the number of non-zero entries
%--------------------------------------------------------------------------
% Inputs:
%   func:   A function handle defines                            (REQUIRED)
%                    (objective, gradient, sub-Hessian)
%   n:      Dimension of the solution x                          (REQUIRED) 
%   lambda: The penalty parameter                                (REQUIRED)  
%   pars  : All parameters are OPTIONAL
%           pars.x0     -- Starting point of x         (default zeros(n,1))
%           pars.tol    -- Tolerance of halting conditions   (default 1e-6)
%           pars.maxit  -- Maximum number of iterations      (default  2e3) 
%           pars.uppf   -- An upper bound of final objective (default -Inf)
%                          Useful for noisy case 
%           pars.eta    -- A positive scalar                    (default 1)  
%                          Tuning it may improve solution quality
%           pars.update -- =1 update penalty parameter lambda   (default 1)
%                          =0 fix penalty parameter lambda
%           pars.disp   -- =1 show results for each step        (default 1)
%                          =0 not show results for each step
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

% Below is one example that you can run
% =========================================================================
% demon sparse linear regression problems 
clc; close all; clear all; addpath(genpath(pwd));

n        = 10000;  
m        = ceil(0.25*n); 
s        = ceil(0.01*n);

Tx       = randperm(n,s);  
xopt     = zeros(n,1);  
xopt(Tx) = (0.25+rand(s,1)).*sign(randn(s,1)); 
A        = randn(m,n)/sqrt(m); 
b        = A*xopt;  
func     = @(x,key,T1,T2)funcLinReg(x,key,T1,T2,A,b);

lambda   = 0.01;
pars.eta = 0.5;
out      = NL0R(func,n,lambda,pars); 
PlotRecovery(xopt,out.sol,[900,500,500,250],1)