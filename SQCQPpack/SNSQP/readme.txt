Please run "demonXXXX.m" to solve different problems 

This package offers 1 solver for sparse quadratically constrained quadratic programming  
based on the algorithm proposed in the following paper: 

SNSQP----------------------------------------------------------------------
    Shuai Li, Shenglong Zhou, Ziyan Luo, 
    Sparse Quadratically Constrained Quadratic Programming via Semismooth Newton Method, 
    https://arxiv.org/abs/2503.15109, 2025.


Please credit it if you use the code for your research.

===========================================================================
function Out = SNSQP(n,s,Q0,q0,Qi,qi,ci,ineqA,ineqb,eqA,eqb,lb,ub,pars)
% This code aims at solving the sparse QCQP in the form of
%
%         min  (1/2)(x'{Q_0}x)+q_0'x 
%         s.t. (1/2)x'*Qi{i}*x+qi(:,i)'*x+ci(i)<=0, i = 1,...,k
%                                 ineqA*x-ineqb<=0
%                                      eqA*x-eqb=0
%                                        lb<=x<=ub
%                                       ||x||_0<=s
%
% where Qi = {Qi{1},...,Qi{k}}, Qi{i} \in R^{n-by-n}, qi \in R^{n-by-k},  ci \in R^{k}
%       ineqA \in R^{m1-by-n},  ineqb \in R^{m1} 
%       eqA   \in R^{m2-by-n},  eqb   \in R^{m2}
%       s << n
%---------------------------------------------------------------------------------------------------           
% Inputs:
%     n:      Dimension of the solution x                                             (required)
%     s:      Sparsity level of x, an integer between 1 and n-1                       (required)
%     Q0:     The quadratic objective matrix in R^{n-by-n}                            (required)        
%     q0:     The quadratic objective vector in R^n                                   (required)
%     Qi:     The quadratic constraint matrix                                         (optional) 
%             MUST be a cell array or [], each entry is matrix in R^{n-by-n}           
%     qi:     The quadratic constraint vector. MUST be a matrix in R^{n-by-k} or []   (optional)           
%     ci:     The quadratic constraint constant in R, must be a vector or []          (optional)
%     ineqA:  The linear inequality constraint matrix in R^{m1-by-n}   or []          (optional)
%     ineqb:  The linear inequality constraint vector in R^{m1}        or []          (optional)
%     eqA:    The linear equality constraint matrix in R^{m2-by-n}     or []          (optional)
%     eqb:    The linear equality constraint vector in R^{m2}          or []          (optional)
%     lb:     The lower bound of x                                                    (optional)
%     ub:     The upper bound of x                                                    (optional)
%             NOTE: 0 must in [lb ub]
%     pars:   Parameters are all OPTIONAL
%             pars.x0       -- Initial point of x                                     (default zeros(n,1))
%             pars.dualquad -- Initial point of mu for quadratic constraints          (default zeros(k,1))
%             pars.dualineq -- Initial point of dual variable for linear inequalities (default zeros(m1,1))
%             pars.dualeq   -- Initial point of dual variable for linear equalities   (default zeros(m2,1))
%             pars.dualbd   -- Initial point of nu  for bound/box constraints         (default zeros(n,1))
%             pars.tau      -- A positive scalar                                      (default 1)
%                              NOTE: tuning a proper tau may yield better solutions     
%             pars.itlser   -- Maximum nonumber of line search                        (default 5)
%             pars.itmax    -- Maximum nonumber of iteration                          (default 10000)
%             pars.show     -- Results shown at each iteration if pars.show=1         (default 1)
%                              Results not shown at each iteration if pars.show=0
%             pars.tol      -- Tolerance of the halting condition                     (default 1e-6)
%
% Outputs:
%     Out.sol:           The sparse solution x
%     Out.sparsity:      Sparsity level of Out.sol
%     Out.error:         Error used to terminate this solver
%     Out.time           CPU time
%     Out.iter:          Number of iterations
%     Out.obj:           Objective function value at Out.sol
%---------------------------------------------------------------------------------------------------
% Send your comments and suggestions to <<< 24110488@bjtu.edu.cn / slzhou2021@163.com >>>
% Warning: Accuracy may not be guaranteed !!!!! 
%---------------------------------------------------------------------------------------------------

% Below is one example that you can run
% =========================================================================
% demon sparse portfolio selection (SPS) problems
clc; clear all; close all;  addpath(genpath(pwd));

n     = 1000;
s     = 10;

B     = 0.01 * rand(ceil(n/4),n);
D     = diag(0.01*rand(n,1));
Q0    = 2*( B'*B + D);
q0    = zeros(n,1); 
Qi    = cell(1,1);
Qi{1} = 2*D;
qi    = zeros(n,1);
ci    = -0.001;
ineqA = -0.5*randn(1,n);
ineqb = -0.002;
eqA   = ones(1,n);
eqb   = 1;
lb    = 0;
ub    = 0.3;
    
pars.x0       = ((lb+ub)/2).*ones(n,1);
pars.tau      = 1; % decrease this value if the algorithm do not converge
pars.dualquad = 0*ones(length(ci));
pars.dualineq = 0.001*ones(length(ineqb)); 
pars.dualeq   = 0.001*ones(length(eqb));
Out           = SNSQP(n,s,Q0,q0,Qi,qi,ci,ineqA,ineqb,eqA,eqb,lb,ub,pars);