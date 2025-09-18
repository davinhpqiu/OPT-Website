clc; close all; clear; addpath(genpath(pwd));

n          = 2000;          % Signal dimension 
m          = ceil(0.5*n);   % Number of measurements
s          = ceil(0.01*n);  % Sparsity level
nf         = 0.05;          % Noisy ratio
r          = 0.02;          % Flipping ratio
k          = ceil(r*m);  

[A,b,bo,xo]= random1bcs('Ind',m,n,s,r,nf); % or 'Cor' 
solver     = {'GPSP','NM01'};
out        = OBCSpack(A,b,s,k,solver{1});  

fprintf(' Time:                  %6.3f sec\n',out.time);
fprintf(' Absolue error:         %6.2f %%\n', norm(xo-out.sol)*100);
fprintf(' Signal-to-noise ratio: %6.2f\n',-10*log10(norm(xo-out.sol)^2));
fprintf(' Hamming distence:      %6.3f\n',nnz(sign(A*out.sol)-b)/m)
fprintf(' Hamming error:         %6.3f\n',nnz(sign(A*out.sol)-bo)/m)
PlotRecovery(xo,out.sol,[1000 450 500 250],1) 

 