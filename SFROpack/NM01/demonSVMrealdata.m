% Solving support vector machine using real datasets
clc; close all; clear all;  addpath(genpath(pwd));

test     = 1;
data     = {'arce','fabc'};
prob     = data{test};
samp     = load(strcat(prob, '.mat')); 
label    = load(strcat(prob, '_class.mat')); 
A        = normalization(samp.X,2);
c        = label.y;
c(c~=1)  = -1;   
[m,n0]   = size(A); 

func     = @(x,key)funcSVM(x,key,1e-4,A,c);
B        = (-c).*[A ones(m,1)];
b        = ones(m,1);
pars.tau = 1; 
lam      = 10;
out      = NM01(func, B, b, lam, pars); 
acc      = 1-nnz( sign([A ones(m,1)]*out.sol)-c )/m;
fprintf(' Training  Size:       %d x %d\n', m,n0);
fprintf(' Training  Time:       %5.3fsec\n',out.time);
fprintf(' Training  Accuracy:   %5.2f%%\n', acc*100) 
fprintf(' Training  Objective:  %5.3e\n',   out.obj);
