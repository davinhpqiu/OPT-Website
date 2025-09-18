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
