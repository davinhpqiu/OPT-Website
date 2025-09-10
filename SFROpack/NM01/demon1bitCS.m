% Solving 1 bit compressive sensing using randomly generated data 
clc; close all; clear all; addpath(genpath(pwd)); %#ok<CLALL>


n            = 1000; 
m            = ceil(0.5*n);
s            = ceil(0.01*n);                      % sparsity level
r            = 0.01;                              % flipping ratio
nf           = 0.05;                              % noisy ratio
[A,c,co,xo]  = random1bcs('Ind',m,n,s,nf,r,0.5);  % data generation

func         = @(x,key)func1BCS(x,key,1e-5,0.5,A,c);
B            = (-c).*A;
b            = (n*8e-5)*ones(m,1);
lam          = 1;
pars.tau     = 1;  
pars.strict  =(n<=2000); 
out          = NM01(func, B, b, lam, pars); 
x            = refine(out.sol,s,A,c);

PlotRecovery(xo,x,[950,500,500,250],1)
fprintf(' Computational time:    %.3fsec\n',out.time);
fprintf(' Signal-to-noise ratio: %.2f\n',-20*log10(norm(x-xo)));
fprintf(' Hamming distance:      %.3f\n',nnz(sign(A*x)-c)/m)
fprintf(' Hamming error:         %.3f\n',nnz(sign(A*x)-co)/m)