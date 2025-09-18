clc; close all; clear; addpath(genpath(pwd));

n      = 2000;                             % Signal dimension 
m      = ceil(0.5*n);                      % Number of measurements
s      = ceil(0.01*n);                     % Sparsity level
nf     = 0.05;                             % Noisy ratio
r      = 0.02;                             % Flipping ratio
k      = ceil(r*m);

A      = randn(m,n);
T      = randperm(n,s);
xo     = zeros(n,1);                      
xo(T)  = (1+rand(s,1)).*sign(randn(s,1));  
xo(T)  = xo(T)/norm(xo(T));                % True sparse solution
bo     = sign(A(:,T)*xo(T)+nf*randn(m,1));
h      = ones(m,1);                        % Flipping vector
T      = randperm(m,k); 
h(T)   = -h(T);
b      = bo.*h; 

solver = {'GPSP','NM01'};
out    = OBCSpack(A,b,s,k,solver{1});  
fprintf(' Time:                  %6.3f sec\n',out.time);
fprintf(' Absolue error:         %6.2f %%\n', norm(xo-out.sol)*100);
fprintf(' Signal-to-noise ratio: %6.2f\n',-10*log10(norm(xo-out.sol)^2));
fprintf(' Hamming distence:      %6.3f\n',nnz(sign(A*out.sol)-b)/m)
fprintf(' Hamming error:         %6.3f\n',nnz(sign(A*out.sol)-bo)/m)
PlotRecovery(xo,out.sol,[1000 450 500 250],1) 
