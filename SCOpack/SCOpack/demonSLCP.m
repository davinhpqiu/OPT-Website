% demon sparse linear complementarity problems
clc; clear; close all; addpath(genpath(pwd));
                   
n         = 10000;  
s         = ceil(0.01*n);
examp     = 2; %= 1, 2, 3
mattype   = {'z-mat','sdp','sdp-non'};
data      = generationLCPdata(mattype{examp},n,s);
func      = @(x,key,T1,T2)funcLCP(x,key,T1,T2,data);

pars.eta  = 1 + 4*(n<=1000); % useful for 'NHTP'
pars.neg  = 1;               % useful for 'IIHT'
pars.tol  = 1e-6;
solver    = {'NHTP', 'GPNP','IIHT'};
out       = SCOpack(func,n,s,solver{2},pars);

fprintf(' Objective:         %5.2e\n',  out.obj);
fprintf(' CPU time:          %.3fsec\n',  out.time);
fprintf(' Sample size:       %dx%d\n', n,n);
if  isfield(data,'xopt')
    PlotRecovery(data.xopt,out.sol,[900,500,500,250],1);
end
