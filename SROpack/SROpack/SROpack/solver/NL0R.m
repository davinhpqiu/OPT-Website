function out = NL0R(func,n,lambda,pars)
%--------------------------------------------------------------------------
% This code aims at solving the L0 norm regularized optimization 
%
%         min_{x\in R^n} f(x) + lambda*||x||_0^0
%
% where f: R^n->R, lambda>0
% ||x||_0^0 counts the number of non-zero entries
%--------------------------------------------------------------------------
% Inputs:
%   func:   A function handle defines                            (REQUIRED)
%                    (objective, gradient, sub-Hessian)
%   n:      Dimension of the solution x                          (REQUIRED) 
%   lambda: The penalty parameter                                (REQUIRED)  
%   pars  : All parameters are OPTIONAL
%           pars.x0     -- Starting point of x         (default zeros(n,1))
%           pars.tol    -- Tolerance of halting conditions   (default 1e-6)
%           pars.maxit  -- Maximum number of iterations      (default  2e3) 
%           pars.uppf   -- An upper bound of final objective (default -Inf)
%                          Useful for noisy case 
%           pars.eta    -- A positive scalar                    (default 1)  
%                          Tuning it may improve solution quality
%           pars.update -- =1 update penalty parameter lambda   (default 1)
%                          =0 fix penalty parameter lambda
%           pars.disp   -- =1 show results for each step        (default 1)
%                          =0 not show results for each step
%--------------------------------------------------------------------------
% Outputs:
%   out.sol :  The sparse solution x
%   out.obj :  Objective function value at out.sol 
%   out.iter:  Number of iterations
%   out.time:  CPU time
%--------------------------------------------------------------------------
% Send your comments and suggestions to <<< slzhou2021@163.com >>>   
% WARNING: Accuracy may not be guaranteed!!!!!  
%--------------------------------------------------------------------------
warning off;

t0 = tic;
if nargin < 3
   fprintf(' No enough inputs. No problems will be solverd!'); return;
elseif nargin<4 
   pars = [];
end 
[x0,eta,tol,maxit,disp,update,uppf,rate] = setparameters(n,pars);  
x       = x0;
Err     = zeros(1,maxit);
Obj     = zeros(1,maxit);
Nzx     = zeros(1,maxit);
TMP     = zeros(3,1);
FNorm   = @(var)norm(var,'fro')^2; 
Funcfg  = @(var)func(var,'fg',[],[]);
FuncH   = @(var,T,J)func(var,'h',T,J);

% Initial check for the starting point
[obj,g] = Funcfg(x);
if FNorm(g)==0 
   fprintf('Starting point is a good stationary point, stop !!!\n'); 
   out.sol = x;
   out.obj = obj;
   return; 
end

if  max(isnan(g))
    x       = zeros(n,1);
    rind    = randi(n);
    x(rind) = rand;
    [obj,g] = Funcfg(x);
end

t   = 0;
xtg = abs(x0-eta*g);
while t < 20
    T   = find(xtg>sqrt(2*eta*lambda));     
    nT  = nnz(T);
    if nT == 0 
       lambda = lambda/1.25;
    elseif nT > 0.12*n
       lambda = lambda*1.25;
    else
        break;
    end  
    t = t+1; 
end
maxlam  = max(abs(g))^2/eta/2;
nx      = 0;
pcgit   = 5;
pcgtol  = 1e-5;
beta    = 0.5;
sigma   = 5e-5;
delta   = 1e-10;
T0      = [];  

if  disp 
    fprintf(' Start to run the solver -- NL0R \n');
    fprintf(' --------------------------------------------------------\n');
    fprintf('  Iter      Error       Objective    Sparsity    Time(sec)\n'); 
    fprintf(' --------------------------------------------------------\n');
end

% The main body  
for iter  = 1:maxit
    
    x0    = x;  
    xtg   = x0-eta*g ; 
    nT    = 0;
    
    while 1
        T     = find(abs(xtg)>sqrt(2*eta*lambda));     
        nT    = nnz(T);
        if nT > 0; break; end
        lambda= lambda/1.05; 
    end
    
    if  iter>1 && nT-nnz(T0)>=0  && nT-nnz(T0)<=5 && Err(iter-1) < tol 
        lambda = lambda0;
        while 1
            T     = find(abs(xtg)>sqrt(2*eta*lambda));     
            nT    = nnz(T);
            if nT > 0; break; end
            lambda= lambda/1.05; 
        end   
    end
    
    if iter>0 && nT > max(0.12,0.2/log2(1+iter))*n 
       Tnew  = SparseApprox(xtg(T),T); 
       nTnew = nnz(Tnew); 
       if ~isempty(Tnew) && nT/nTnew < 20 && nT~=nTnew
           T  = Tnew;  
           nT = nTnew;
       end 
    end
     
    TTc   = setdiff(T,T0);
    flag  = isempty(TTc);    

    % Calculate the error for stopping criteria 
    FxT       = sqrt(FNorm(g(T))+FNorm(x(TTc)));
    Err(iter) = FxT/sqrt(n);
    Nzx(iter) = nx;
    if  disp  && (mod(iter,10)==0 ||iter < 100)
        fprintf(' %4d      %8.2e     %9.2e      %4d      %5.3fsec\n',...
        iter, FxT, obj, nx,toc(t0)); 
    end
    
    % Stopping criteria   
    TMP    = [TMP(2:end); obj];  
    stop1  = Err(iter)<tol && std(TMP)<1e-8*(1+abs(obj));
    stop1  = stop1 && nx==nT && flag;  
    stop2  = iter > 3 && obj < uppf && nx<=ceil(n/4);
    stop3  = norm(g)<tol && nx<=ceil(n/4); 
    if  iter>1 && (stop1 || stop2 || stop3)  
        if  disp  && ~(mod(iter,10)==0 ||iter < 100)
            fprintf(' %4d      %8.2e     %9.2e      %4d      %5.3fsec\n',...
            iter, FxT, obj, nx,toc(t0)); 
        end
        break;   
    end
   
    % update next iterate
    if  iter   == 1 || flag    % two consective iterates have same supports
        H       = FuncH(x0,T,[]);     
        if isa(H,'function_handle')
           d    = my_cg(H,-g(T),pcgtol,pcgit,zeros(nT,1)); 
        else 
           d    = -H\g(T);  
        end
       
        dg     = sum(d.*g(T)); 
        ngT    = FNorm(g(T));  
        if dg  > max(-delta*FNorm(d), -ngT) || max(isnan(dg)) 
        d      = -g(T); 
        dg     = ngT; 
        end
    else                  % two consective iterates have different supports                       
        [H,D]   = FuncH(x0,T,TTc);   
        if  isa(D,'function_handle')
            rhs =  D(x0(TTc))-g(T);  
        else
            rhs = D*x0(TTc) - g(T); 
        end
        if  isa(H,'function_handle')
            d   = my_cg(H, rhs,pcgtol,pcgit,zeros(nT,1));  
        else 
            d   = H\rhs; 
        end
         
        Fnz     = FNorm(x(TTc))/4/eta;
        dgT     = sum(d.*g(T));
        dg      = dgT-sum(x0(TTc).*g(TTc));
        
        delta0  = delta;
        if Fnz  > 1e-4; delta0 = 1e-4; end
 
        ngT     = FNorm(g(T));
        if dgT  > max(-delta0*FNorm(d)+Fnz, -ngT) || isnan(dg) 
           d    = -g(T); 
           dg   = ngT; 
        end            
    end
    
    % Armijo line search
    alpha      = 1; 
    x          = zeros(n,1);    
    obj0       = obj;             
    for i      = 1:6
        x(T)   = x0(T) + alpha*d;
        obj    = Funcfg(x);
        if obj < obj0  + alpha*sigma*dg; break; end        
        alpha  = beta*alpha;
    end
 
    T0       = T; 
    [obj,g]  = Funcfg(x);
    Obj(iter)= obj; 
    
%   Update tau    
    if  mod(iter,10)==0  
        OBJ = Obj(iter-9:iter);
        if Err(iter)>1/iter^2 || sum(OBJ(2:end)>1.5*OBJ(1:end-1))>=2 
            if iter<1500; eta = eta/1.25; 
            else;         eta = eta/1.5; 
            end     
        else          
            eta = eta*1.25;   
        end
    end 
    
%   Update lambda    
    nx  = nnz(x); 
    if  iter>5 && (nx > 2*max(Nzx(1:iter-1))) && Err(iter)<1e-2
        rate0   = 2/rate;   
        x       = x0;
        nx      = nnz(x0); 
        nx0     = Nzx(iter-1);  
        [obj,g] = Funcfg(x); 
        rate    = 1.1;
    else  
        rate0   = rate; 
    end
       
    if exist('nx0') && nx < nx0
       rate0 = 1;   
    end
 
    lambda0 = lambda;
    if mod(iter,1)==0 && update 
       lambda  = min(maxlam,lambda*(2*(nx>=0.1*n)+rate0));  
    end

end

%Results output ------------------------------------------------- 
[obj,g]     = Funcfg(x);
time        = toc(t0);
out.sparsity= nnz(x); 
out.time    = time;
out.iter    = iter;
out.sol     = x;
out.obj     = obj; 
out.Obj     = Obj;

if disp 
   fprintf(' --------------------------------------------------------\n');
   normgrad    = norm(g);
   if normgrad < 1e-6
      fprintf(' A global optimal solution might be found\n');
      fprintf(' because of ||gradient|| = %5.2e!\n',normgrad); 
      fprintf(' --------------------------------------------------------\n');
   end
end

end


% Setup parameters --------------------------------------------------------
function  [x0,eta,tol,maxit,disp,update,uppf,rate] = setparameters(n,pars)  
    rate0 = (n<=1e3)*0.5+(n>1e3)/exp(3/log10(n));
    if isfield(pars,'x0');    x0    = pars.x0;    else; x0 = zeros(n,1);end
    if isfield(pars,'eta');   eta   = pars.eta;   else; eta   = 1;      end
    if isfield(pars,'rate');  rate  = pars.rate;  else; rate  = rate0;  end
    if isfield(pars,'disp');  disp  = pars.disp;  else; disp  = 1;      end 
    if isfield(pars,'maxit'); maxit = pars.maxit; else; maxit = 2000;   end
    if isfield(pars,'uppf');  uppf  = pars.uppf;  else; uppf  = -Inf;   end 
    if isfield(pars,'tol');   tol   = pars.tol;   else; tol   = 1e-6;   end 
    if isfield(pars,'update');update= pars.update;else; update= 1;      end
end

% get the sparse approximation of x ---------------------------------------
function T = SparseApprox(x0,T0)
x       = abs(x0); 
sx      = sort(x(x~=0));  
if  length(sx)<=2
    th  = sx(end);
else
    [mx,it] = max(normalize(sx(2:end)./sx(1:end-1)));
    th      = 0; 
    if mx   > 10 && it(1)>1
       th   = sx(it(1)); 
    end
end       
T = T0(x>th); 
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