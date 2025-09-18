function out = NM01(A,b,sp,pars)
% This code aims at solving the one-bit compressive sensing with form
%
%      min  sum_{i=1}^n (x_i^2+vareps)^{q/2} + lam * sum_{i=1}^m l_{0/1}(epsilon-b_i<A_i,x>)
%
% vareps > 0, 1 > q > 0, lam > 0, b > 0, epsilon>0
% l_{0/1}(t) = 1 if t > 0 and 0 if t <= 0
% b_i is the i-th entry of b
% A_i is the i-th row of A
% =========================================================================
% Inputs:
%     A   : The sample data in \R^{m-by-n}                       (REQUIRED)
%     b   : The binary observation \in R^m, b_i\in{-1,1}         (REQUIRED)    
%     sp  : Sparsity of the ture signal if known                 (REQUIRED)
%           Set it as [] if unknown                               
%     pars: Parameters are all optional                          (OPTIONAL)
%           pars.x0      --  The initial point         (default zeros(n,1))
%           pars.vareps  --  Parameter in the objective       (default 0.5)
%           pars.q       --  Parameter in the objective       (default 0.5)
%           pars.epsilon --  Parameter in the objective      (default 0.15)
%           pars.lam     --  The penalty parameter              (default 1)
%           pars.tau     --  A useful parameter                 (default 1) 
%           pars.maxit   --  Maximum number of iterations    (default 1000)  
% =========================================================================
% Outputs:
%     Out.x:      The solution 
%     Out.obj:    The objective function value
%     Out.time    CPU time
%     Out.iter:   Number of iterations
% =========================================================================
% Written by Shenglong Zhou on 27/06/2021 based on the algorithm proposed in
%     Shenglong Zhou, Lili Pan, Naihua Xiu, Houduo Qi, SIOPT,  
%     Quadratic convergence of smoothing Newton's method for 0/1 loss optimization,
%     SIAM Journal on Optimization, 31(4): 3184–3211, 2021 
% Send your comments and suggestions to <<< slzhou2021@163.com >>>                                  
% WARNING: Accuracy may not be guaranteed!!!!!  
% =========================================================================
warning off; 
t0  = tic; 
if  nargin < 3  
    disp(' Inputs is not enough !!! \n');
    return;
elseif nargin < 4
    pars = []; 
end

[m,n]  = size(A);
negb   = -b;
if  n  < 1e4
    Ab  = negb.*A;
else
    Ab  = spdiags(negb,0,m,m)*A;    
end

[maxit,lam,tau,mu,epsilon,vareps,cgtol,x,disp] = GetParameters(m,n,pars); 
lam         = max(lam, epsilon^2/2/tau);  
[grad,hess] = def_func(pars);
Fnorm       = @(var)norm(var,'fro')^2;
z           = ones(m,1);    
ACC         = zeros(maxit,1); 
if  norm(x) == 0
    Ax      = epsilon*ones(m,1);
else
    Ax      = Ab*x + epsilon;    
end
Axz         = Ax;   
maxAcc      = 0; 
 
if  disp
    fprintf('\n Start to run the solver: NM01\n')
    fprintf('------------------------------------------\n');
    fprintf('  Iter           HamDist          CPUTime \n')
    fprintf('------------------------------------------\n');
end
for iter      = 1:maxit
     
    [T,empT]  = Ttau(Axz,Ax,tau,lam); 
    g         = grad(x,vareps);
    nT        = nnz(T);  
    
    if  empT
        tmp1 = g;
        raw  = 0; 
        err  = Fnorm(g) + raw  + Fnorm(z);
    else      
        P    = Ab(T,:);
        zT   = z(T);
        tmp1 = g + (zT'*P)'; 
        tmp2 = Ax(T);
        raw  = Fnorm(tmp2);     
        err  = Fnorm(tmp1) + raw +  (Fnorm(z) -Fnorm(zT) ); 
    end
        
    sb        = sign(negb.*(Ax-epsilon)); 
    sb(sb==0) = -1;
    acc       = 1-nnz(sb+negb)/m; 
    x0        = x;
    if iter   > 1 && acc<min(0.5,ACC(iter-1))
       acc    = 1-acc; 
       x0     = -x;
    end
    
    ACC(iter)    = acc; 
    if ACC(iter) > maxAcc
       maxAcc    = ACC(iter);
       maxx      = x0;  
    end
    
    stop1 = (iter>5 && acc<ACC(iter-1) && min(ACC(iter-2:iter-1))==1);    
    stop2 = (iter>5 && acc>0.99995 && raw < 1e-5*sqrt(n) && n>500); 

    if ~stop1 && disp 
        fprintf('  %3d          %8.2f           %.3fsec\n',iter,acc*100,toc(t0));  
    end
    
    if (err<1e-4 && stop2) || stop1 ; break;  end  
     
    if  empT
        u      = - g;
        v      = - z;
    else                
        H      = hess(x,vareps); 
         
        if  nT  < n
            H1  = 1./H; 
            rhs = tmp2 - P*(H1.*tmp1);
            if  nT  < 2e3          
                D   = P*(H1.*P');            
                D(1:nT+1:end) = D(1:nT+1:end) + mu; 
                vT  = D\rhs;  
            else    
                fx  = @(var)(mu*var + P*(H1.*(var'*P)'));
                vT  = my_cg(fx,rhs,cgtol,30,zeros(nT,1)); 
            end        
            v   = -z; 
            v(T)= vT;  
            u   = - H1.*(tmp1 + (vT'*P)');      
        else
            rhs = -mu*tmp1-(tmp2'*P)';
            if  n   < 2e3 
                D   = P'*P;  
                D(1:n+1:end) = D(1:n+1:end) + mu*H';  
                u   = D\rhs;  
            else
                fx = @(var)(mu*(H.*var) + ((P*var)'*P)');  
                u  = my_cg(fx,rhs,cgtol,50,zeros(n,1)); 
            end
            v    = -z; 
            v(T) = (tmp2+P*u)/mu;  
        end
    end 
    
    x   = x + u; 
    z   = z + v;  
    Ax  = Ab*x + epsilon; 
      
    Axz = Ax + tau*z;  
    vareps = max(1e-5,vareps/2);
    if mod(iter,5)==0   
       mu  = max(1e-10,mu/2);                                                                          
    end    
end

if  disp
    fprintf('------------------------------------------\n');
end
if acc < ACC(1); x = maxx; end 
out.lam  = z;
out.time = toc(t0);
out.iter = iter;

if ~isempty(sp) % refinement step
    K       = 6;
    [sx,Ts] = maxk(abs(x),sp+K-1);
    HD      = ones(1,K);
    X       = zeros(n,K); 
    if sx(sp)-sx(sp+1) <= 2e-4 
        tem    = Ts(sp:end);
        for i  = 1:K
            X(:,i)          = zeros(n,1);
            X(Ts(1:sp-1),i) = x(Ts(1:sp-1));
            X(tem(i),i)     = x(tem(i));
            X(:,i)          = X(:,i)/norm(X(:,i));
            HD(i)           = nnz(sign(A*X(:,i))-b)/m; 
        end
        [~,i]   = min(HD); 
        out.sol = X(:,i); 
    else
        out.sol           = zeros(n,1);  
        out.sol(Ts(1:sp)) = x(Ts(1:sp))/norm(x(Ts(1:sp))); 
    end   
else
  x       = SparseApprox(x); 
  out.sol = x / norm(x);   
end
 
clear A b A0 B0 P
end

%--------------------------------------------------------------------------
function [maxit,lam,tau,mu,epsilon,vareps,tolcg,x0,disp] = GetParameters(m,n,pars)
    maxit   = 1e3;
    lam     = 1;
    tau     = 1; 
    mu      = 0.05;   
    epsilon = 0.15; 
    vareps  = 0.5; 
    tolcg   = 1e-10*sqrt(max(m,n));
    x0      = zeros(n,1);
    disp  = 1;
    if isfield(pars,'maxit');   maxit   = pars.maxit;    end 
    if isfield(pars,'lam') ;    lam     = pars.lam;      end 
    if isfield(pars,'tau');     tau     = pars.tau;      end
    if isfield(pars,'mu');      mu      = pars.mu;       end
    if isfield(pars,'epsilon'); epsilon = pars.epsilon;  end
    if isfield(pars,'vareps');  vareps  = pars.vareps;   end
    if isfield(pars,'x0');      x0      = pars.x0;       end 
    if isfield(pars,'disp');    disp   = pars.disp;      end
end

%select the index set T----------------------------------------------------
function [T,empT] = Ttau(Axz,Ax,tau,lam)
    tl   = sqrt(tau*lam/2);
    T    = find(abs(Axz-tl)<tl); 
    empT = isempty(T);  
    if  empT
        zp   = Ax(Ax>=0);  
        T    = [];
        if nnz(zp)>0
            s   = ceil(0.01*nnz(zp));   
            tau = (zp(s))^2/2/lam;     
            tl  = sqrt(tau*lam/2);
            T   = find(abs(Ax-tl)<tl);          
        end
        empT = isempty(T); 
    end
end

% Define functions---------------------------------------------------------
function [grad,hess] = def_func(pars)
    %f(x) = (x^2+e)^(q/2)    
    q    = 0.5;     
    if isfield(pars,'q'); q = pars.q;  end 
    q1   = q/2-1;
    q2   = q/2-2;
    q3   = q-1;
    grad = @(t,e)( t.* ((t.*t+e).^q1) );             
    hess = @(t,e)( ((t.*t+e).^q2).*(q3*t.*t+e) ); 
end

% Conjugate gradient method-------------------------------------------------
function x = my_cg(fx,b,cgtol,cgit,x)
    r = b;
    e = sum(r.*r);
    t = e;
    for i = 1:cgit  
        if e < cgtol*t; break; end
        if  i == 1  
            p = r;
        else
            p = r + (e/e0)*p;
        end
        w  = fx(p); 
        a  = e/sum(p.*w);
        x  = x + a * p;
        r  = r - a * w;
        e0 = e;
        e  = sum(r.*r);
    end
    
end

% get the sparse approximation of x----------------------------------------
function sx = SparseApprox(x)
    n       = length(x);
    xo      = x;
    x       = x.*x;
    T       = find(x>1e-2/n);
    [sx,id] = sort(x(T),'descend'); 
    y       = 0;
    nx      = sum(x(T)); 
    nT      = nnz(T);
    t       = zeros(nT-1,1);
    for i   = 1:nT
        if y > 0.5*nx; break; end
        y    = y + sx(i); 
        if i < nT
        t(i) = sx(i)/sx(i+1);
        end
    end
    
    if  i  < nT
        j  = find(t==max(t)); 
        i  = j(1); 
    else
        i  = nT;
    end
    
    if i  > 1
        i = min(nT,10*i);
    else
        i = nT;
    end
    sx = zeros(n,1);
    sx(T(id(1:i))) = xo(T(id(1:i)));
end
