Please run "demonXXXX.m" to solve different problems 

This package offers 3 solvers for sparsity constrained optimization problems  
based on algorithms proposed in the following 3 papers: 

NHTP-----------------------------------------------------------------------
    Shenglong Zhou, Naihua Xiu, and Huoduo Qi, 
    Global and quadratic convergence of Newton hard-thresholding pursuit, 
    Journal of Machine Learning Research, 22, 1−45, 2021.

GPNP-----------------------------------------------------------------------
    Shenglong Zhou, 
    Gradient projection newton pursuit for sparsity constrained optimization, 
    Applied and Computational Harmonic Analysis, 61, 75-100, 2022.

IIHT-----------------------------------------------------------------------
    Lili Pan, Shenglong Zhou, Naihua Xiu, and Huoduo Qi, 
    A convergent iterative hard thresholding for sparsity and nonnegativity constrained optimization, 
    Pacific Journal of Optimization, 13, 325-353, 2017.

Please credit them if you use the code for your research.

===========================================================================
function out = SCOpack(func,n,s,solvername,pars)
%--------------------------------------------------------------------------
% This code aims at solving the sparsity constrained optimization (SCO),
%
%         min_{x\in R^n} f(x),  s.t. ||x||_0<=s
%
% or sparsity and non-negative constrained optimization (SNCO):
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
%                          =1 for model (SNCO)
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
% demon a simple sparsity constrained problem
clc; close all; clear all;  addpath(genpath(pwd));

n        = 2;
s        = 1; 
func     = @funcSimpleEx;
solver   = {'NHTP','GPNP','IIHT'};
pars.eta = 0.1; % useful for 'NHTP'
out      = SCOpack(func,n,s,solver{2},pars); 

fprintf(' Objective:      %.4f\n', out.obj); 
fprintf(' CPU time:      %.3fsec\n', out.time);