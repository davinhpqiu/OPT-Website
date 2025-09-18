% demon compressive sensing problems 
clc; close all; clear all; addpath(genpath(pwd));

n        = 1000;  
m        = ceil(0.25*n); 
s        = ceil(0.05*n);       
nf       = 0.00;

Tx       = randperm(n,s);  
xopt     = zeros(n,1);  
xopt(Tx) = (0.25+rand(s,1)).*sign(randn(s,1)); 
A        = randn(m,n); 
data.A   = A/(issparse(A)*log(m)+~issparse(A)*sqrt(m));
data.b   = data.A*xopt+nf*randn(m,1);  
func     = @(x,key,T1,T2)funcCS(x,key,T1,T2,data);

pars.eta = 1; 
lambda   = 0.01;   
out      = NL0R(func,n,lambda, pars); 

fprintf(' CPU time:          %.3fsec\n',  out.time);
fprintf(' Objective:         %5.2e\n',  out.obj);
fprintf(' True Objective:    %5.2e\n',  norm(data.A*xopt-data.b)^2/2);
fprintf(' Sample size:       %dx%d\n', m,n);
PlotRecovery(xopt,out.sol,[900,500,500,250],1)
