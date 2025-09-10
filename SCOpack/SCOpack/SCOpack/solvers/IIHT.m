function out = IIHT(func,n,s, pars)
%--------------------------------------------------------------------------
% This code aims at solving the sparsity constrained optimization (SCO),
%
%         min_{x\in R^n} f(x),  s.t. ||x||_0<=s
%
% or non-negative and sparsity constrained optimization (NSCO):
%
%         min_{x\in R^n} f(x),  s.t. ||x||_0<=s, x>=0 
%
% where f: R^n->R and s<<n is an integer. 
%--------------------------------------------------------------------------
% Inputs:
%     func: A function handle defines                            (REQUIRED)
%                    (objective,gradient,sub-Hessain)
%     n   : Dimension of the solution x                          (REQUIRED)
%     s   : Sparsity level of x, an integer between 1 and n-1    (REQUIRED)
%     pars: Parameters are all OPTIONAL
%           pars.x0    --  Starting point of x         (default zeros(n,1))
%           pars.neg   --  =0 for model (SCO)                   (default 1)
%                          =1 for model (NSCO)
%           pars.disp  --  =1 show results for each step       (default 1)
%                          =0 not show results for each step
%           pars.maxit --  Maximum number of iterations      (default 2000) 
%           pars.tol   --  Tolerance of stopping criteria    (default 1e-4)
%           pars.uppf  --  An upper bound of final objective (default -Inf)
%                          Useful for noisy case
% Outputs:
%     out.sol :   The sparse solution x
%     out.obj :   Objective function value at out.sol 
%     out.iter:   Number of iterations
%     out.time:   CPU time
%--------------------------------------------------------------------------
% Send your comments and suggestions to <<< slzhou2021@163.com >>>                                  
% WARNING: Accuracy may not be guaranteed!!!!!  
%--------------------------------------------------------------------------
warning off;
t0 = tic;
if nargin<3; fprintf('Imputs are not enough!\n'); return; end
if nargin<4; pars = []; end
if isfield(pars,'disp');  disp  = pars.disp;  else; disp  = 1;          end
if isfield(pars,'maxit'); maxit = pars.maxit; else; maxit = 5e3;        end
if isfield(pars,'tol');   tol   = pars.tol;   else; tol = 1e-6*sqrt(n); end  
if isfield(pars,'neg');   neg   = pars.neg;   else; neg   = 0;          end  
if isfield(pars,'uppf');  uppf  = pars.uppf;  else; uppf  = -Inf;       end 
func   = @(var)func(var,'fg',[],[]); 
Fnorm  = @(var)norm(var,'fro')^2;
sigma0 = 1e-4;
x      = zeros(n,1);
xo     = zeros(n,1);
OBJ    = zeros(5,1);
% main body
if  disp 
    fprintf(' Start to run the sover -- IIHT \n'); 
    fprintf(' -------------------------------------------\n');
    fprintf('  Iter     Error      Objective       Time \n'); 
    fprintf(' -------------------------------------------\n');
end
[f,g]    = func(x);
scale    = (max(f,norm(g))>n); 
scal     = n*(scale==1)+(scale==0); 
fs       = f/scal;  
gs       = g/scal;  
for iter = 1:maxit     
    
    x_old  = x;      
    % Line search for setp size alpha
    fx_old = fs;
    alpha  = log(1+iter);
    for j  = 1:10
        tp = x_old-alpha*gs;
        if   neg
             tp     = max(0, tp);   
             [mx,T] = maxk(tp,s);
        else
             [mx,T] = maxk(tp,s,'ComparisonMethod','abs');    
        end
        x      = xo; 
        x(T)   = mx;
        fs     = func(x)/scal; 
        if fs  < fx_old-0.5*sigma0*Fnorm(x-x_old); break; end
        alpha  = alpha/2;        
    end
 
    [f,g]  = func(x);
    fs     = f/scal;  
    gs     = g/scal;  
    OBJ    = [OBJ(2:end); fs];
    % Stop criteria 
	error  = scal*norm(gs(T))/max(1,norm(mx)); 
    normg  = norm(g);
    if disp && (iter<=10 || mod(iter,10)==0)
       fprintf(' %4d     %5.2e    %9.2e     %5.3fsec\n',iter,error,fs*scal,toc(t0)); 
    end
 
    stop1 = error < tol && (std(OBJ) < 1e-8*(1+abs(fs))); 
    stop2 = normg < tol ;
    stop3 = fs*scal < uppf;     
	if  iter > 1 && (stop1 || stop2 || stop3)
        if disp && ~(iter<=10 || mod(iter,10)==0)
          fprintf(' %4d     %5.2e    %9.2e     %5.3fsec\n',iter,error,fs*scal,toc(t0)); 
        end
        break; 
    end  

end

if  disp
    fprintf(' -------------------------------------------\n');
end

out.sol   = x;
out.obj   = fs*scal;
out.iter  = iter;
out.time  = toc(t0);
out.error = error; 
if  normg < 1e-5 && disp
    fprintf(' A global optimal solution might be found\n');
    fprintf(' because of ||gradient||=%5.2e!\n',normg);  
    fprintf(' -------------------------------------------\n');
end
end
