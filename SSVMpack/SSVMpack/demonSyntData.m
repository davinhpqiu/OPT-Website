% demon randomly generated data
clc; close all; clear all;  addpath(genpath(pwd));
 
type    = 1;   % 1 or 2 or 3
Ex      = {'2D', '3D', 'nD'};
m0      = 4e2;  
n0      = 100;
[Atrain,ytrain,Atest,ytest] = randomData(Ex{type},m0,n0,0);  
[m, n]  = size(Atrain); 

t       = 1;
solver  = {'NM01','NSSVM'};
pars.C  = 0.25;
out     = SSVMpack(Atrain,ytrain,solver{t},pars);  
acc     = accuracy(Atrain,out.w,ytrain); 
tacc    = accuracy(Atest,out.w,ytest);
fprintf(' Training  Time:             %5.3fsec\n',out.time);
fprintf(' Training  Size:             %dx%d\n',m,n);
fprintf(' Training  Accuracy:         %5.2f%%\n', acc*100) 
fprintf(' Testing   Size:             %dx%d\n',size(Atest,1),n);
fprintf(' Testing   Accuracy:         %5.2f%%\n',tacc*100);
fprintf(' Number of Support Vectors:  %d\n',out.sv); 
if isequal(Ex{type},'2D') && m <400
   figure('Renderer', 'painters', 'Position', [900, 400, 300 300])
   plot2D(Atrain,ytrain,out.w,solver{t},acc); xlabel('Training data')
   figure('Renderer', 'painters', 'Position', [1220, 400, 300 300])
   plot2D(Atest,ytest,out.w,solver{t},tacc); xlabel('Testing data')
end
