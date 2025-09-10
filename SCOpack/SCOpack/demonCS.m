% demon compressive sensing problems 
clc; close all; clear all; addpath(genpath(pwd));

n        = 10000;  
m        = ceil(0.25*n); 
s        = ceil(0.025*n);       
nf       = 0.00;

Tx       = randperm(n,s);  
xopt     = zeros(n,1);  
xopt(Tx) = randn(s,1); 
A        = randn(m,n); 
data.A   = A/(issparse(A)*log(m)+~issparse(A)*sqrt(m));
data.b   = data.A*xopt+nf*randn(m,1);  

func     = @(x,key,T1,T2)funcCS(x,key,T1,T2,data);
if    nf > 0; pars.eta = 0.5; end    % useful for 'NHTP'
pars.tol = 1e-6;
solver   = {'NHTP','GPNP','IIHT'};
out      = SCOpack(func,n,s,solver{2},pars); 

fprintf(' CPU time:          %.3fsec\n',  out.time);
fprintf(' Objective:         %5.2e\n',  out.obj);
fprintf(' True Objective:    %5.2e\n',  norm(data.A*xopt-data.b)^2/2);
fprintf(' Sample size:       %dx%d\n', m,n);
PlotRecovery(xopt,out.sol,[900,500,500,250],1)
