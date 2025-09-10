Please run "demonXXXX.m" to solve different problems 

This package offers 1 solver for step function regularized optimization problems  
based on the algorithm proposed in the following paper: 

NM01-----------------------------------------------------------------------
    Shenglong Zhou, Lili Pan, Naihua Xiu, Houduo Qi, 
    Quadratic convergence of smoothing Newton's method for 0/1 loss optimization,
    SIAM Journal on Optimization, 31(4): 3184–3211, 2021.

Please give credits to this paper if you use the code for your research.

===========================================================================
function out = NM01(func,B,b,lam,pars)
% -------------------------------------------------------------------------
% This code aims at solving the support vector machine with form
%
%      min  f(x) + lam * ||(Bx+b)_+||_0
%
% where f is twice continuously differentiable
% lam > 0, B\in\R^{m x n}, b\in\R^{m x 1}
% (z)_+ = (max{0,z_1},...,max{0,z_m})^T
% ||(z)_+ ||_0 counts the number of positive entries of z
% -------------------------------------------------------------------------
% Inputs:
%   func: A function handle defines (objective,gradient,Hessain) (REQUIRED)
%   B   : A matrix \R^{m x n}                                    (REQUIRED)      
%   b   : A vector \R^{m x 1}                                    (REQUIRED)
%   lam : The penalty parameter                                  (REQUIRED)
%   pars: Parameters are all OPTIONAL
%         pars.x0     -- The initial point             (default zeros(n,1))
%         pars.tau    -- A useful paramter                   (default 1.00)
%         pars.mu0    -- A smoothing parameter               (default 0.01)
%         pars.maxit  -- Maximum number of iterations        (default 1000)  
%         pars.tol    -- Tolerance of halting conditions   (1e-7*sqrt(n*m)) 
%         pars.strict -- = 0, loosely meets halting conditions  (default 0)
%                        = 1, strictly meets halting conditions  
%                        pars.strict=1 is useful for low dimensions  
% -------------------------------------------------------------------------
% Outputs:
%   out.sol:  The solution 
%   out.obj:  The objective function value
%   out.time: CPU time
%   out.iter: Number of iterations
% -------------------------------------------------------------------------
% Send your comments and suggestions to <<< slzhou2021@163.com >>>    
% WARNING: Accuracy may not be guaranteed!!!!!  
% -------------------------------------------------------------------------

% Below is one example that you can run
% =========================================================================
% Solving support vector machine using four synthetic samples
clc; close all; clear all;  addpath(genpath(pwd));

a           = 10;
A           = [0 0; 0 1; 1 0; 1 a]; 
c           = [-1 -1  1  1]';
[m,n]       = size(A);  

func        = @(x,key)funcSVM(x,key,1e-4,A,c);
B           = (-c).*[A ones(m,1)];
b           = ones(m,1);
lam         = 10;
pars.tau    = 1;
pars.strict = 1;
out         = NM01(func, B, b, lam, pars); 
x           = out.sol;        

figure('Renderer', 'painters', 'Position', [1000, 300,350 330])
axes('Position', [0.08 0.08 0.88 0.88] );
scatter([1;1],[0 a],80,'+','m'), hold on
scatter([0;0],[0,1],80,'x','b'), hold on
line([-x(3)/x(1) -x(3)/x(1)],[-1 1.1*a],'Color', 'r')
axis([-.1 1.1 -1 1.1*a]),box on,grid on
ld = strcat('NM01:',num2str(func(x,'a')*100,'%.0f%%'));
legend('Positive','Negative',ld,'location','NorthWest')
