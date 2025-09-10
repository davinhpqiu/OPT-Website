function out = GPNP(func,n,s,pars)
%--------------------------------------------------------------------------
% This code aims at solving the sparsity constrained optimization,
%
%         min_{x\in R^n} f(x)  s.t.  \|x\|_0<=s
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
%           pars.eta   --  A positive scalar for 'NHTP'         (default 1)  
%                          Tuning it may improve solution quality 
%           pars.disp  --  =1, show results for each step       (default 1)
%                          =0, not show results for each step
%           pars.maxit --  Maximum number of iterations      (default 2000) 
%           pars.tol   --  Tolerance of stopping criteria    (default 1e-6)
%           pars.uppf  --  An upper bound of final objective (default -Inf)
%                          Useful for noisy case
% Outputs:
%     Out.sol :   The sparse solution x
%     Out.time:   CPU time
%     Out.iter:   Number of iterations
%     Out.obj :   Objective function value at Out.sol 
%--------------------------------------------------------------------------
% Send your comments and suggestions to <<< slzhou2021@163.com >>>                                  
% WARNING: Accuracy may not be guaranteed!!!!!  
%--------------------------------------------------------------------------

warning off;
t0 = tic;

if nargin<3
    fprintf(' Error!!!\n Inputs are not enough!!!\n'); return; 
elseif nargin<4
    pars=[]; 
end

[sigma,J,flag,alpha0,gamma,thd,disp,tol,uppf,maxit]...
          = set_parameters(n,s,pars);
x         = zeros(n,1);
xo        = zeros(n,1);

FNorm     = @(var)norm(var,'fro')^2;
Funcfg    = @(var)func(var,'fg',[],[]);
FuncH     = @(var,T,J)func(var,'h',T,J);
[fx,gx]   = Funcfg(x);
[~,Tx]    = maxk(gx,s,'ComparisonMethod','abs');
Tx        = sort(Tx);  
minobj    = zeros(maxit,1);
minobj(1) = fx;
OBJ       = zeros(5,1);

% main body
if  disp 
    fprintf(' Start to run the solver -- GPNP \n');
    fprintf(' -------------------------------------------\n');
    fprintf('  Iter     Error      Objective       Time \n'); 
    fprintf(' -------------------------------------------\n');
end
 
for iter = 1:maxit     
     
    % Line search for setp size alpha
 
    alpha  = alpha0;  
    for j  = 1:J
        [subu,Tu] = maxk(x-alpha*gx,s,'ComparisonMethod','abs');
        u         = xo; 
        u(Tu)     = subu;  
        fu        = Funcfg(u);  
        if fu     < fx - sigma*FNorm(u-x); break; end
        alpha     = alpha*gamma;        
    end

    [fu,gx] = Funcfg(u);
    normg   = FNorm(gx);
    x       = u;
    fx      = fu; 
    
    % Newton step
    sT    = sort(Tu); 
    mark  = nnz(sT-Tx)==0;
    Tx    = sT;
    eps   = 1e-4;
    if ( mark || normg < 1e-4 || alpha0==1 ) && s<=5e4
        v = xo; 
        H = FuncH(u,Tu,[]); 
        if s     < 200 && ~isa(H,'function_handle')  
           subv  = subu + H\(-gx(Tu)); 
           eps   = 1e-8;
        else     
           cgit  = min(25,5*iter);   
           subv  = subu + my_cg(H,-gx(Tu),1e-10*n,cgit,zeros(s,1));
        end  
        v(Tu)    = subv;  
        [fv,gv]  = Funcfg(v); 
        if fv   <= fu  - sigma * FNorm(subu-subv)
           x     = v;  
           fx    = fv;
           subu  = subv;  
           gx    = gv;
           normg = FNorm(gx);  
        end   
    end
    
    % Stop criteria  
    error     = sqrt(FNorm(gx(Tu))); 
    obj       = fx;
    OBJ       = [OBJ(2:end); obj];
    if  disp && (iter<=10 || mod(iter,10)==0)
        fprintf(' %4d     %5.2e    %9.2e     %5.3fsec\n',iter,error,fx,toc(t0)); 
    end

    maxg       = max(abs(gx));
    minx       = min(abs(subu));
    J          = 8;
    if error^2 < tol*1e3 && normg>1e-2 && iter < maxit-10
       J       = min(8,max(1,ceil(maxg/minx)-1));     
    end  

    if isfield(pars,'uppf') && obj<=uppf && flag
        maxit  = iter + 100*s/n; 
        flag   = 0; 
    end
 
    minobj(iter+1) = min(minobj(iter),fx);  
    if fx    < minobj(iter) 
        xmin = x;  
        fmin = fx;  
    end
    
    if iter  > thd 
       count = std(minobj(iter-thd:iter+1) )<1e-10;
    else
       count = 0; 
    end
 
    stop1 = error < tol && (std(OBJ) < eps*(1+abs(obj))); 
    stop2 = sqrt(normg) < tol;
    stop3 = fx < uppf;  
    if  iter > 1 && (stop1 || stop2 || stop3 || count)
        if  count && fmin < fx; x=xmin; fx=fmin; end
        if  disp && ~(iter<=10 || mod(iter,10)==0)
           fprintf(' %4d     %5.2e    %9.2e     %5.3fsec\n',iter,error,fx,toc(t0)); 
        end
        break; 
    end  
    
end

out.sol   = x;
out.obj   = fx;
out.iter  = iter;
out.error = error;
out.time  = toc(t0);

if  disp
    fprintf(' -------------------------------------------\n')
end
if  normg < 1e-10 && disp
    fprintf(' A global optimal solution may be found\n');
    fprintf(' because of ||gradient|| = %5.3e!\n',sqrt(normg)); 
    fprintf(' -------------------------------------------\n')
end

end

% set parameters-------------------------------------------------------
function [sigma,J,flag,alpha0,gamma,thd,disp,tol,uppf,maxit]=set_parameters(n,s,pars)
    sigma     = 1e-8; 
    J         = 1;    
    flag      = 1;
    alpha0    = 5;
    gamma     = 0.5;
    if s/n   <= 0.05 && n >= 1e4
       alpha0 = 1; 
       gamma  = 0.1;  
    end
    if s/n   <= 0.05
       thd    = ceil(log2(2+s)*50); 
    else
        if  n    > 1e3 
            thd  = 100;
        elseif n > 500
            thd  = 500;
        else
            thd  = ceil(log2(2+s)*750);
        end
    end      
    if isfield(pars,'disp');  disp  = pars.disp;  else; disp  = 1;     end
    if isfield(pars,'tol');   tol   = pars.tol;   else; tol   = 1e-6;  end  
    if isfield(pars,'uppf');  uppf  = pars.uppf;  else; uppf  = -Inf;  end 
    if isfield(pars,'maxit'); maxit = pars.maxit; else; maxit = 1e4;   end
 
end

% conjugate gradient-------------------------------------------------------
function x = my_cg(fx,b,cgtol,cgit,x)
    if norm(b,'fro')==0; x=zeros(size(x)); return; end
    if ~isa(fx,'function_handle'); fx = @(v)fx*v; end
    r = b;
    if nnz(x)>0; r = b - fx(x);  end
    e = norm(r,'fro')^2;
    t = e;
    p = r;
    for i = 1:cgit  
        if e < cgtol*t; break; end
        w  = fx(p);
        pw = p.*w;
        a  = e/sum(pw(:));
        x  = x + a * p;
        r  = r - a * w;
        e0 = e;
        e  = norm(r,'fro')^2;
        p  = r + (e/e0)*p;
    end 
end
