% demon a simple sparsity constrained problem
clc; close all; clear all;  addpath(genpath(pwd));

n        = 2;
s        = 1; 
func     = @funcSimpleEx;
solver   = {'NHTP','GPNP','IIHT'};
pars.eta = 0.1; % useful for 'NHTP'
out      = SCOpack(func,n,s,solver{2},pars); 

fprintf(' Objective:      %.4f\n', out.obj); 
fprintf(' CPU time:      %.3fsec\n', out.time);
fprintf(' Iterations:        %4d\n', out.iter);