% Classifiy 4 points with 1 outlier 
clc; close all; clear all; addpath(genpath(pwd));

a        = 10;
A        = [0 0; 0 1; 1 0; 1 a]; 
y        = [-1 -1  1  1]';
[m,n]    = size(A);  
pars.C   = 1; 
out      = SSVMpack(A,y,'NM01',pars);    

figure('Renderer', 'painters', 'Position', [1000, 300,350 330])
axes('Position', [0.08 0.08 0.88 0.88] );
scatter([1;1],[0 a],80,'+','m'), hold on
scatter([0;0],[0,1],80,'x','b'), hold on
line([-out.w(3)/out.w(1) -out.w(3)/out.w(1)],[-1 1.1*a],'Color', 'r')
axis([-.1 1.1 -1 1.1*a]),box on, grid on
ld = strcat('NM01:',num2str(accuracy(A,out.w,y)*100,'%.0f%%'));
legend('Positive','Negative',ld,'location','NorthWest')
