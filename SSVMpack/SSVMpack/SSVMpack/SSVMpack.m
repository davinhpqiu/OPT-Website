function out = SSVMpack(A,y,solver,pars)
% -------------------------------------------------------------------------
% This package aims to solve the binary classification problems
% Inputs:
%  A:       The smaple matrix \in R^{m-by-n},                    (REQUIRED)
%  y:       The binary label \in R^m, b_i\in{-1,1}               (REQUIRED)    
%  solver:  A text string, can be one of {'NM01','NSSVM'}        (REQUIRED)            
%  pars:    Parameters are optional                              (OPTIONAL) 
%           -------------  For NSSVM --------------------------------------
%           pars.alpha --  Starting point in \R^m       (default zeros(m,1))
%           pars.s0    --  The initial sparsity    (default n(log(m/n))^2))
%           pars.C     --  A positive scalar in (0,1]        (default  1/4)  
%           pars.c     --  A positive scalar in (0,1]        (default 1/40)  
%           pars.tune  --  Tune the sparsity level              
%                          Do not tune the sparsity level       (default 0)
%           pars.maxit --  Maximum number of iterations      (default 1000) 
%           pars.tol   --  Tolerance of the halting criteria (default 1e-4) 
%           pars.disp  --  Display results for each step        (default 1)  
%                          Do not display results for each step 
%           -------------  NM01 -------------------------------------------
%           pars.x0    --  The initial point           (default zeros(n,1))
%           pars.C     --  The penalty parameter                (default 1)
%           pars.tau   --  A useful paramter                    (default 5)
%           pars.maxit --  Maximum number of iterations      (default 1000)  
%           pars.tol   --  Tolerance of the halting criteria (default 1e-4) 
%           pars.disp  --  Display results for each step        (default 1)  
%                          Do not display results for each step 
% -------------------------------------------------------------------------
% Outputs:
%     out.w:      The solution of the primal problem, i.e., the classifier
%     out.sv:     Number of support vectors 
%     out.time    CPU time
%     out.iter:   Number of iterations
%     Out.acc:    Classification accuracy
% -------------------------------------------------------------------------
% Send your comments and suggestions to <slzhou2021@163.com> 
% Warning: Accuracy may not be guaranteed !!!!!! 
% -------------------------------------------------------------------------

warning off; 
if  nargin<3  
    disp(' Inputs are not enough !!! \n');
    return;
elseif nargin<4
    pars = []; 
end
[m,n] = size(A);

if m < 0.1*n
   disp(' Suggest solver <NM01> !!! \n'); 
   solver = 'NM01';
end

switch solver     
    case 'NSSVM'  
        out = NSSVM(A,y,pars);
    case 'NM01' 
        if isfield(pars,'C')
           pars.lam = max(0.1,pars.C); 
        end
        out = NM01(A,y,pars);        
end
end

