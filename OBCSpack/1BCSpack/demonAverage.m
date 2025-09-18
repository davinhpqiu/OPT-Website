clc; clear all; close all; addpath(genpath(pwd));

n    = 500;           % Signal dimension 
m    = ceil(0.5*n);   % Number of measurements
s    = ceil(0.01*n);  % Signal sparsity
r    = 0.05;          % Flipping ratio
k    = ceil(r*m);     % Upper bound of sign flips
v    = 0.5;           % Correlation parameter

Type = 'Ind';         % or 'Cor' 
test = 's';           % change 'test' to see effect of GPSP to 
                      % factors {'s','m','r','v','n'}
switch test
  case 'm',   test0 = linspace(0.2,2,10);   
  case 's',   test0 = 2:10;  
  case 'r',   test0 = 0.01:0.01:0.1;  
  case 'v',   test0 = 0.1:0.1:0.9; Type = 'Cor';
  case 'n',   test0 = (5:5:20)*1e3;
end

f         = isequal(test,'n');  
S         = 20*(1-f)+10*f;
recd      = zeros(nnz(test0),4);
pars.disp = 0;
for j  = 1:nnz(test0)
    switch test
      case 'm',   m = ceil(test0(j)*n);
      case 's',   s = test0(j);
      case 'r',   r = test0(j);
      case 'v',   v = test0(j);    
      case 'n',   n = test0(j); s=ceil(0.01*n); m=ceil(n/2);
    end

    for ii = 1 : S    
        % Generate data 
        [A,b,bo,xo] = random1bcs(Type,m,n,s,r,0.01,v);
        out         = OBCSpack(A,b,s,k,'GPSP',pars);       
        recd(j,1)   = recd(j,1) - 20*log10(norm(xo-out.sol));      
        recd(j,2)   = recd(j,2) + nnz(sign(A*out.sol)-b)/m;
        recd(j,3)   = recd(j,3) + nnz(sign(A*out.sol)-bo)/m;
        recd(j,4)   = recd(j,4) + out.time;
    end
end

recd = recd/S; 
close all;
figure('Renderer', 'painters', 'Position', [600, 600, 900 200])
ylab  = {'SNR','HD','HE','TIME'};
xloc  = [ -0.09  -0.045   0.00  0.04];
for j = 1:4
    sub  = subplot(1,4,j); 
    pos  = get(sub, 'Position'); 
    plot(test0,recd(:,j),'b.-','LineWidth',0.75), hold on,
    axis([min(test0) max(test0) 0-2*(j==1) 0.35+35*(j==1)-0.346*(j==4)]); 
    grid on, xlabel(test), ylabel(ylab{j}),   
    set(sub, 'Position',pos+[xloc(j),0.05,0.04,-0.05] )
end   
