function out = NM01(func,B,b,lam,pars)
% -------------------------------------------------------------------------
% This code aims at solving the support vector machine with form
%
%      min  f(x) + lam * ||(Bx+b)_+||_0
%
% where f is twice continuously differentiable
% lam > 0, B\in\R^{m x n}, b\in\R^{m x 1}
% (z)_+ = (max{0,z_1},...,max{0,z_m})^T
% ||(z)_+ ||_0 counts the number of positive entries of z
% -------------------------------------------------------------------------
% Inputs:
%   func: A function handle defines (objective,gradient,Hessain) (REQUIRED)
%   B   : A matrix \R^{m x n}                                    (REQUIRED)      
%   b   : A vector \R^{m x 1}                                    (REQUIRED)
%   lam : The penalty parameter                                  (REQUIRED)
%   pars: Parameters are all OPTIONAL
%         pars.x0     -- The initial point             (default zeros(n,1))
%         pars.tau    -- A useful paramter                   (default 1.00)
%         pars.mu0    -- A smoothing parameter               (default 0.01)
%         pars.maxit  -- Maximum number of iterations        (default 1000)  
%         pars.tol    -- Tolerance of halting conditions   (1e-7*sqrt(n*m)) 
%         pars.strict -- = 0, loosely meets halting conditions  (default 0)
%                        = 1, strictly meets halting conditions  
%                        pars.strict=1 is useful for low dimensions  
% -------------------------------------------------------------------------
% Outputs:
%   out.sol:  The solution 
%   out.obj:  The objective function value
%   out.time: CPU time
%   out.iter: Number of iterations
% -------------------------------------------------------------------------
% Send your comments and suggestions to <<< slzhou2021@163.com >>>    
% WARNING: Accuracy may not be guaranteed!!!!!  
% -------------------------------------------------------------------------
warning off;
if  nargin < 4  
    disp(' Inputs are not enough !!! \n');
    return;
elseif nargin < 5
    pars = []; 
end

t0    = tic; 
[m,n] = size(B);
[maxit,tau,mu,x,tol,tolcg,strict] = GetParameters(m,n,pars);

Fnorm = @(var)norm(var,'fro')^2;
obje  = @(var)func(var,'f');
grad  = @(var)func(var,'g');
Hess  = @(var)func(var,'h');
fBxb  = @(var)B*var+b; 

accu  = @(var)func(var,'a');
if isempty(accu(x))
   accu = @(var)(1-nnz(fBxb(var)>0)/m); 
end

z       = ones(m,1);    
u       = x;
Bxb     = fBxb(x);
Bxz     = Bxb+tau*z;    
lam     = max(1/2/tau,lam);   
Acc     = zeros(maxit,1);
Obj     = zeros(maxit,1);
T       = [];
%lam = Initialam(m,n,Axz,tau);
 
fprintf(' Start to run the solver -- NM01\n');
fprintf(' -----------------------------------------------------------\n');
fprintf('  Iter     Accuracy     Objective      Error       Time(sec) \n')
fprintf(' -----------------------------------------------------------\n');

for iter = 1:maxit
    T0           = T;
    [T,empT,lam] = Ttau(Bxz,Bxb,tau,lam); 
    
    if  iter>3 && std(Acc(iter-3:iter-1))<1e-8
        T = unique(union(T0,T));
    end
    
    nT = nnz(T);
    if  nnz(T0)==nT && nnz(T0-T)==0
        flag = 0;  
    else
        flag = 1;
    end
    
    g  = grad(x);      
    if  empT 
        error = Fnorm(g)+Fnorm(z); 
    else
        if  flag  
            BT    = B(T,:); 
            fBT   = @(var)BT*var;
            fBTt  = @(var)(var'*BT)';
        end
        zT    = z(T);
        rhs1  = g + fBTt(zT);  
        rhs2  = Bxb(T);  
        error = Fnorm(rhs1)+Fnorm(rhs2)+Fnorm(z)-Fnorm(z(T));
    end
    
    error     = error/sqrt(n*m);
    Acc(iter) = accu(x)*100;
    Obj(iter) = obje(x);
    if iter <10 || mod(iter,10)==0
    fprintf('  %3d     %8.3f      %8.4f     %8.3e      %.3fsec\n',...
            iter,Acc(iter),Obj(iter),error,toc(t0)); 
    end

    stop    = 0;
    if iter > 5 && strict
       stop = max(error,Fnorm(u))<tol;
    elseif iter>5 && ~strict
       stop1 = min(error,Fnorm(u))<tol; 
       stop2 = std(Acc(iter-5:iter))<1e-10; 
       stop3 = std(Obj(iter-5:iter))<1e-4*(1+sqrt(abs(Obj(iter))));  
       stop  = stop1 || (stop2 && stop3);
    end
        
    if stop
        if iter > 10 && mod(iter,10)~=0
        fprintf('  %3d     %8.3f      %8.4f     %8.3e      %.3fsec\n',...
        iter,Acc(iter),Obj(iter),error,toc(t0)); 
        end
        break; 
    end   
     
    if  empT
        u = - g;
        v = - z;
    else   
        H      = Hess(x);
        isfunH = isa(H, 'function_handle'); 
        if ~isfunH && n <= 1e3 && nT <=1e4
            u    = (BT'*BT+mu*H)\( -mu*rhs1-fBTt(rhs2) ); 
            v    = -z; 
            v(T) = (fBT(u)+rhs2)/mu;  
        elseif ~isfunH && n > 1e3  && isdiag(H) && n<=5e3
            invH = diag(H);
            invH(abs(invH)<1e-8)=1e-4/iter;  
            invH = 1./invH; 
            D    = BT*(invH.*BT'); 
            D(1:nT+1:end)=D(1:nT+1:end)+mu; 
            vT   = D\( rhs2 - fBT(invH.*rhs1) );  
            v    = -z; 
            v(T) = vT; 
            u    = -invH.*(rhs1 + fBTt(vT));      
        else 
            if  isfunH 
                fx = @(var)( fBTt(fBT(var))+ mu*H(var) ); 
            else
                fx = @(var)( fBTt(fBT(var))+ mu*(H*var) );
            end
            u    = my_cg(fx,-mu*rhs1-fBTt(rhs2),tolcg,20,zeros(n,1));  
            v    = -z; 
            v(T) = (fBT(u)+rhs2)/mu;
        end
    end
    
    alpha = 1;
    x0    = x;
    z0    = z;
    obj0  = obje(x);
    for i = 1:4
        x = x0 + alpha*u;     
        if obje(x)<obj0; break; end
        alpha = 0.8*alpha;        
    end
    z   = z0 + alpha*v;
%     x   = x+u; 
%     z   = z+v; 
    Bxb = fBxb(x);   
    if mod(iter,5)==0   
       mu  = max(1e-8,mu/1.1);          
       tau = max(1e-4,tau/1.1);
       lam = lam*1.1;
    end         
    Bxz = Bxb + tau*z; 

end

fprintf(' -----------------------------------------------------------\n');
out.sol  = x; 
out.obj  = obje(x);
out.time = toc(t0);
out.iter = iter;
clear B b 
end

%--------------------------------------------------------------------------
function [maxit,tau,mu,x0,tol,tolcg, strict] = GetParameters(m,n,pars)
    maxit  = 1e3;
    mn     = m*n;
    tolcg  = 1e-8*sqrt(mn);
    tol    = 1e-7*sqrt(mn);
    tau    = 1.00;
    mu     = 0.01; 
    x0     = zeros(n,1); 
    strict = 0;
    if isfield(pars,'maxit');  maxit  = pars.maxit;  end
    if isfield(pars,'tau');    tau    = pars.tau;    end
    if isfield(pars,'x0');     x0     = pars.x0;     end
    if isfield(pars,'tol');    tol    = pars.tol;    end
    if isfield(pars,'mu0');    mu     = pars.mu0;    end
    if isfield(pars,'strict'); strict = pars.strict; end
end


%select the index set T----------------------------------------------------
function [T,empT,lam] = Ttau(Bxz,Bxb,tau,lam)

    tl   = sqrt(tau*lam/2);
    T    = find(abs(Bxz-tl)<=tl);
    empT = isempty(T);

    if  empT
        zp   = Bxb(Bxb>=0);  
        T    = [];
        if  nnz(zp)>0
            s   = ceil(0.01*nnz(zp));   
            tau = (zp(s))^2/2/lam;     
            tl  = sqrt(tau*lam/2);
            T   = find(abs(Bxb-tl)<tl);          
        end
        empT = isempty(T); 
    end
    
end

% Set initial lam----------------------------------------------------------
function  lam  = Initialam(m,n,z,tau)
    zp   = z(z>0);
    s    = min([m,20*n,nnz(zp)]);  
    lam  = max(5,max((zp(s))^2,1)/2/tau);    
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