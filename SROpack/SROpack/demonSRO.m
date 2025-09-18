% demon a simple SRO problem
clc; close all; clear all;  addpath(genpath(pwd));

n        = 2;
lambda   = 0.5;
pars.eta = 0.1;
out      = NL0R(@funcSimpleEx,n,lambda,pars); 
fprintf(' Objective:      %.4f\n', out.obj); 
fprintf(' CPU time:      %.3fsec\n', out.time);
fprintf(' Iterations:        %4d\n', out.iter);