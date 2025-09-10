% Solving support vector machine using four synthetic samples
clc; close all; clear all;  addpath(genpath(pwd));

a           = 10;
A           = [0 0; 0 1; 1 0; 1 a]; 
c           = [-1 -1  1  1]';
[m,n]       = size(A);  

func        = @(x,key)funcSVM(x,key,1e-4,A,c);
B           = (-c).*[A ones(m,1)];
b           = ones(m,1);
lam         = 10;
pars.tau    = 1;
pars.strict = 1;
out         = NM01(func, B, b, lam, pars); 
x           = out.sol;        

figure('Renderer', 'painters', 'Position', [1000, 300,350 330])
axes('Position', [0.08 0.08 0.88 0.88] );
scatter([1;1],[0 a],80,'+','m'), hold on
scatter([0;0],[0,1],80,'x','b'), hold on
line([-x(3)/x(1) -x(3)/x(1)],[-1 1.1*a],'Color', 'r')
axis([-.1 1.1 -1 1.1*a]),box on,grid on
ld = strcat('NM01:',num2str(func(x,'a')*100,'%.0f%%'));
legend('Positive','Negative',ld,'location','NorthWest')
