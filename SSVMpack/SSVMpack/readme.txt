Please run "demonXXXX.m" to solve different problems 

This package offers 2 solvers for sparse support vector machine problems 
based on the algorithms proposed in the following 2 papers: 

NM01-----------------------------------------------------------------------
   Shenglong Zhou, Lili Pan, Naihua Xiu and Huoduo Qi, 
   Quadratic convergence of smoothing Newton's method for 0/1 loss optimization, 
   SIAM Journal on Optimization, 31, 3184–3211, 2021.

NSSVM-----------------------------------------------------------------------
   Shenglong Zhou, 
   Sparse SVM for sufficient data reduction, 
   IEEE Transactions on Pattern Analysis and Machine Intelligence, 44, 5560-5571, 2022.

Please credit them if you use the code for your research.

===========================================================================
function out = SSVMpack(A,y,solver,pars)
% -------------------------------------------------------------------------
% This package aims to solve the binary classification problems
% Inputs:
%  A:       The smaple matrix \in R^{m-by-n},                    (REQUIRED)
%  y:       The binary label \in R^m, b_i\in{-1,1}               (REQUIRED)    
%  solver:  A text string, can be one of {'NM01','NSSVM'}        (REQUIRED)            
%  pars:    Parameters are optional                              (OPTIONAL) 
%           -------------  For NSSVM --------------------------------------
%           pars.alpha --  Starting point in \R^m       (default zeros(m,1))
%           pars.s0    --  The initial sparsity    (default n(log(m/n))^2))
%           pars.C     --  A positive scalar in (0,1]        (default  1/4)  
%           pars.c     --  A positive scalar in (0,1]        (default 1/40)  
%           pars.tune  --  Tune the sparsity level              
%                          Do not tune the sparsity level       (default 0)
%           pars.maxit --  Maximum number of iterations      (default 1000) 
%           pars.tol   --  Tolerance of the halting criteria (default 1e-4) 
%           pars.disp  --  Display results for each step        (default 1)  
%                          Do not display results for each step 
%           -------------  NM01 -------------------------------------------
%           pars.x0    --  The initial point           (default zeros(n,1))
%           pars.C     --  The penalty parameter                (default 1)
%           pars.tau   --  A useful paramter                    (default 5)
%           pars.maxit --  Maximum number of iterations      (default 1000)  
%           pars.tol   --  Tolerance of the halting criteria (default 1e-4) 
%           pars.disp  --  Display results for each step        (default 1)  
%                          Do not display results for each step 
% -------------------------------------------------------------------------
% Outputs:
%     out.w:      The solution of the primal problem, i.e., the classifier
%     out.sv:     Number of support vectors 
%     out.time    CPU time
%     out.iter:   Number of iterations
%     Out.acc:    Classification accuracy
% -------------------------------------------------------------------------
% Send your comments and suggestions to <slzhou2021@163.com> 
% Warning: Accuracy may not be guaranteed !!!!!! 
% -------------------------------------------------------------------------

% Below is one example that you can run
% =========================================================================
% Classifiy 4 points with 1 outlier 
clc; close all; clear all; addpath(genpath(pwd));

a        = 10;
A        = [0 0; 0 1; 1 0; 1 a]; 
y        = [-1 -1  1  1]';
[m,n]    = size(A);  
pars.C   = 1; 
out      = SSVMpack(A,y,'NM01',pars);   

figure('Renderer', 'painters', 'Position', [1000, 300,350 330])
axes('Position', [0.08 0.08 0.88 0.88] );
scatter([1;1],[0 a],80,'+','m'), hold on
scatter([0;0],[0,1],80,'x','b'), hold on
line([-out.w(3)/out.w(1) -out.w(3)/out.w(1)],[-1 1.1*a],'Color', 'r')
axis([-.1 1.1 -1 1.1*a]),box on, grid on
ld = strcat('NM01:',num2str(accuracy(A,out.w,y)*100,'%.0f%%'));
legend('Positive','Negative',ld,'location','NorthWest')