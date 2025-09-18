% demon sparse linear regression problems 
clc; close all; clear all; addpath(genpath(pwd));

n        = 1000;  
m        = ceil(0.25*n); 
s        = ceil(0.05*n);

Tx       = randperm(n,s);  
xopt     = zeros(n,1);  
xopt(Tx) = (0.25+rand(s,1)).*sign(randn(s,1)); 
A        = randn(m,n)/sqrt(m); 
b        = A*xopt;  
func     = @(x,key,T1,T2)funcLinReg(x,key,T1,T2,A,b);

lambda   = 0.01;
pars.eta = 1.0;
out      = NL0R(func,n,lambda,pars); 
PlotRecovery(xopt,out.sol,[900,500,500,250],1)