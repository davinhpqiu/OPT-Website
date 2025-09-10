% demon sparse linear regression problems 
clc; close all; clear all; addpath(genpath(pwd));

n        = 20000;  
m        = ceil(0.25*n); 
s        = ceil(0.025*n);

Tx       = randperm(n,s);  
xopt     = zeros(n,1);  
xopt(Tx) = randn(s,1); 
A        = randn(m,n)/sqrt(m); 
b        = A*xopt;  

func     = @(x,key,T1,T2)funcLinReg(x,key,T1,T2,A,b);
pars.tol = 1e-6;
solver   = {'NHTP','GPNP','IIHT'};
out      = SCOpack(func,n,s,solver{2},pars);
PlotRecovery(xopt,out.sol,[900,500,500,250],1)