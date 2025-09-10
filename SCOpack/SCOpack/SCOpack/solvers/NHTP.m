function out = NHTP(func,n,s,pars)
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
%           pars.eta   --  A positive scalar for 'NHTP'        (default, 1)  
%                          Tuning it may improve solution quality 
%           pars.disp  --  =1, show results for each step       (default,1)
%                          =0, not show results for each step
%           pars.maxit --  Maximum number of iterations      (default,2000) 
%           pars.tol   --  Tolerance of stopping criteria    (default,1e-6)
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
t0  = tic;
if  nargin<3
    fprintf(' No enough inputs. No problems will be solverd!'); return;
elseif nargin < 4  
    pars = [];
end 

FNorm   = @(var)norm(var,'fro')^2;
Funcfg  = @(var)func(var,'fg',[],[]);
FuncH   = @(var,T,J)func(var,'h',T,J);

[x0,eta,disp,maxit,tol,uppf] = getparameters(n,s,Funcfg,pars);  
x       = x0;
beta    = 0.5;
sigma   = 5e-5;
delta   = 1e-10;
pcgtol  = 0.1*tol*s;

T0      = [];
Error   = zeros(1,maxit);
Obj     = zeros(1,maxit);
OBJ     = zeros(5,1);
xo      = zeros(n,1);
[obj,g] = Funcfg(x0);
if  disp 
    fprintf(' Start to run the solver -- NHTP \n');
    fprintf(' -------------------------------------------\n');
    fprintf('  Iter     Error      Objective       Time \n'); 
    fprintf(' -------------------------------------------\n');
end

% Initial check for the starting point
if  FNorm(g)<1e-20 && nnz(x)<=s
    fprintf(' Starting point is a good solution. Stop NHTP\n'); 
    out.sol  = x;
    out.obj  = obj;
    out.time = toc(t0);
    return;
end

if  max(isnan(g))
    x0      = zeros(n,1);
    rind    = randi(n);
    x0(rind)= rand;
    [obj,g] = Funcfg(x0);
end
 
% The main body  
for iter = 1:maxit
     
    xtg   = x0-eta*g;
    [~,T] = maxk(x0-eta*g,s,'ComparisonMethod','abs');
    T     = sort(T);
    TTc   = setdiff(T0,T);
    flag  = isempty(TTc);    
    gT    = g(T);
    
    % Calculate the error for stopping criteria   
    xtaus           = max(0,max(abs(g))-min(abs(x(T)))/eta); 
    if  flag
        FxT         = sqrt(FNorm(gT));
        Error(iter) = xtaus + FxT;
    else
        FxT         = sqrt(FNorm(gT)+ abs(FNorm(x)-FNorm(x(T))) );
        Error(iter) = xtaus + FxT;    
    end 
    
    %Error(iter)  = FNorm(gT);
    if  disp && (iter<=10 || mod(iter,10)==0)
        fprintf(' %4d     %5.2e    %9.2e     %5.3fsec\n',iter,Error(iter),obj,toc(t0)); 
    end
             
    % Stopping criteria
    OBJ   = [OBJ(2:end); obj];
    stop1 = Error(iter)<tol && (std(OBJ)<1e-8*(1+abs(obj)));  
    stop2 = sqrt(FNorm(g))<tol;
    stop3 = obj<uppf;   
    if  iter > 1 && (stop1 || stop2 || stop3)
        if  disp && ~(iter<=10 || mod(iter,10)==0)
            fprintf(' %4d     %5.2e    %9.2e     %5.3fsec\n',iter,Error(iter),obj,toc(t0)); 
        end 
        break;  
    end
 
    % update next iterate
    if  iter   == 1 || flag           % update next iterate if T==supp(x^k)   c
        H       =  FuncH(x0,T,[]); 
        if  isa(H,'function_handle')
            d   = my_cg(H,-gT,pcgtol,25,zeros(s,1));
        else
            d   = H\(-gT); 
        end
        dg      = sum(d.*gT);
        ngT     = FNorm(gT);
        if  dg  > max(-delta*FNorm(d), -ngT) || isnan(dg)
            d   = -gT; 
            dg  = ngT; 
        end
    else                              % update next iterate if T~=supp(x^k) 
        [H,D]   = FuncH(x0,T,TTc);  
        if  isa(D,'function_handle')
            rhs = D(x0(TTc))-gT;
        else
            rhs = D*x0(TTc)-gT; 
        end
        if  isa(H,'function_handle')
            d   = my_cg(H,rhs,pcgtol,25,zeros(s,1)); 
        else       
            d   = H\rhs; 
        end              
        Fnz     = FNorm(x(TTc))/4/eta;
        dgT     = sum(d.*gT);
        dg      = dgT-sum(x0(TTc).*g(TTc));
        
        delta0  = delta;
        if Fnz  > 1e-4; delta0 = 1e-4; end
        ngT     = FNorm(gT);
        if dgT  > max(-delta0*FNorm(d)+Fnz, -ngT) || isnan(dg) 
            d   = -gT; 
            dg  = ngT; 
        end
    end
    
    alpha    = 1; 
    x        = xo;    
    obj0     = obj;        
    Obj(iter)= obj;
    % Amijio line search
    for i      = 1:6
        x(T)   = x0(T) + alpha*d;
        obj    = Funcfg(x);
        if obj < obj0 + alpha*sigma*dg; break; end        
        alpha  = beta*alpha;
    end
 
    % Hard Thresholding Pursuit if the obj increases
    fhtp    = 0;
    if obj  > obj0 
       x(T) = xtg(T); 
       obj  = Funcfg(x); 
       fhtp = 1;
    end
    
    % Stopping criteria
    flag1   = (abs(obj-obj0)<1e-6*(1+abs(obj)) && fhtp);    
    flag2   = (abs(obj-obj0)<1e-8*(1+abs(obj))&& Error(iter)<1e-2); 
    if  iter>10 &&  (flag1 || flag2)      
        if obj > obj0
           iter    = iter-1; 
           x       = x0; 
           T       = T0; 
        end   
        fprintf(' %4d     %5.2e    %9.2e     %5.3fsec\n',iter,Error(iter),obj,toc(t0)); 
        break;
     end 
 
    T0      = T; 
    x0      = x; 
    [obj,g] = Funcfg(x);   
    
    % Update eta
    if  mod(iter,50)==0  
        if  Error(iter)>1/iter^2  
            if iter < 1500
                eta = eta/1.25; 
            else          
                eta = eta/1.5; 
            end     
        else
            eta = eta*1.15; 
        end
    end     
    
    
end

% results output
out.time    = toc(t0);
out.iter    = iter;
out.sol     = x;
out.obj     = obj; 

if disp 
   normgrad = sqrt(FNorm(g)); 
   fprintf(' -------------------------------------------\n')
   if normgrad<1e-5
      fprintf(' A global optimal solution might be found\n');
      fprintf(' because of ||gradient|| = %5.2e!\n', normgrad); 
      if out.iter>1500
      fprintf('\n Since the number of iterations reaches to %d\n',out.iter);
      fprintf(' Try to rerun the solver with setting a smaller pars.eta \n'); 
      end
      fprintf(' -------------------------------------------\n')
   end
end
end

% initialize parameters ---------------------------------------------------
function [x0,eta,disp,maxit,tol,uppf]=getparameters(n,s,func,pars)

    if isfield(pars,'disp');  disp  = pars.disp;  else; disp  = 1;      end 
    if isfield(pars,'maxit'); maxit = pars.maxit; else; maxit = 2000;   end
    if isfield(pars,'tol');   tol   = pars.tol;   else; tol   = 1e-6;   end   
    if isfield(pars,'x0');    x0    = pars.x0;    else; x0 = zeros(n,1);end 
    if isfield(pars,'uppf');  uppf  = pars.uppf;  else; uppf  = -Inf;   end 
    
    if isfield(pars,'eta')      
        eta    = pars.eta;       
    else % set a proper parameter eta
        [~,g1] = func(ones(n,1)) ;
        abg1   = abs(g1);
        T      = find(abg1>1e-8);
        maxe   = sum(1./(abg1(T)+eps))/nnz(T);
        if  isempty(T) 
            eta = 10*(1+s/n)/min(10, log(n));
        else
            if maxe>2
                eta  = (log2(1+maxe)/log2(maxe))*exp((s/n)^(1/3));
            elseif maxe<1
                eta  = (log2(1+ maxe))*(n/s)^(1/2);    
            else
                eta  = (log2(1+ maxe))*exp((s/n)^(1/3));
            end     
        end
    end   
 
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
