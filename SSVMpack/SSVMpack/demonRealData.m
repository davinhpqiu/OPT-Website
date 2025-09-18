% demon real data 
clc; close all; clear all; addpath(genpath(pwd));
  
load dhrb.mat;  
load dhrbclass.mat;  

[m0,n]    = size(A);         
A         = normalization(A,2); % data normalization 
m         = ceil(0.9*m0);       % data splition 
Train     = randperm(m0,m); 
Ttest     = setdiff(1:m0,Train); 
Atrain    = A(Train,:);     
Atest     = A(Ttest,:);
ytrain    = y(Train,:);     
ytest     = y(Ttest);    

t         = 1;
pars.C    = 0.25;
solver    = {'NM01','NSSVM'};
out       = SSVMpack(Atrain,ytrain,solver{t},pars);
acc       = accuracy(Atrain,out.w,ytrain);
tacc      = accuracy(Atest,out.w,ytest);

fprintf(' Training  Time:             %5.3fsec\n',out.time);
fprintf(' Training  Size:             %dx%d\n',size(Atrain,1),size(Atrain,2));
fprintf(' Training  Accuracy:         %5.2f%%\n', acc*100);
fprintf(' Testing   Size:             %dx%d\n',size(Atest,1),size(Atest,2));
fprintf(' Testing   Accuracy:         %5.2f%%\n',tacc*100);
fprintf(' Number of Support Vectors:  %d\n',out.sv); 