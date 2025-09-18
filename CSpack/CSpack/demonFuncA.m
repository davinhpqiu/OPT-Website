% demon compressed sensing problems with a function handle matrix 
clc; clear; close all; addpath(genpath(pwd));

n       = 1000;  
m       = ceil(0.25*n); 
s       = ceil(0.05*n); 

T       = randperm(n,s);  
xopt    = zeros(n,1);
xopt(T) = (0.1+rand(s,1)).*sign(randn(s,1));  
A       = randn(m,n);   
A       = A/(issparse(A)*log(m)+~issparse(A)*sqrt(m));
b       = A(:,T)*xopt(T)+0.000*randn(m,1);  
At      = @(var)A'*var; % At is a function handle
A       = @(var)A*var;  % A  is a function handle

solver  = {'GPNP','NHTP','IIHT','NL0R','PSNP','MIRL1'};
out     = CSpack(A,At,b,n,s,solver{1}); 

fprintf(' Objective at xopt:       %.2e\n', norm(A(xopt)-b)^2/2);
fprintf(' Objective at out.sol:    %.2e\n', out.obj);
fprintf(' Sparsity of out.sol:     %2d\n', nnz(out.sol));
fprintf(' Computational time:      %.3fsec\n',out.time); 
PlotRecovery(xopt,out.sol,[1000 500 500 250],1);
