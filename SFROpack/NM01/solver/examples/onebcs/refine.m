function refx = refine(x,sp,A0,c)
% x:  a vector in R^{n x 1}               (REQUIRED)
% sp: a positive integer in [1,n] or [].  (REQUIRED)
% AO: a matrix in R^{m x n}. If sp ~=[], then A0 is REQUIRED
% c:  a vector in R^{m x 1}. If sp ~=[], then c  is REQUIRED
 
    [m,n] = size(A0); 
    if ~isempty(sp) % refinement step
        K       = 6;
        [sx,Ts] = maxk(abs(x),sp+K-1);  
        HD      = ones(1,K);
        X       = zeros(n,K); 
        if sx(sp)-sx(sp+1) <= 5e-2 
            tem    = Ts(sp:end);
            for i  = 1:K
                X(:,i)          = zeros(n,1);
                X(Ts(1:sp-1),i) = x(Ts(1:sp-1));
                X(tem(i),i)     = x(tem(i));
                X(:,i)          = X(:,i)/norm(X(:,i));
                HD(i)           = nnz(sign(A0*X(:,i))-c)/m; 
            end
            [~,i] = min(HD); 
            refx  = X(:,i); 
        else
            refx           = zeros(n,1);   
            refx(Ts(1:sp)) = x(Ts(1:sp))/norm(x(Ts(1:sp))); 
        end   
    else
        refx  = SparseApprox(x); 
        refx  = refx / norm(refx);   
    end
    
    if max(isnan(refx))==1
       refx  = SparseApprox(x); 
       refx  = refx / norm(refx); 
    end
    
end

% get the sparse approximation of x----------------------------------------
function x0 = SparseApprox(x0)
    n       = length(x0); 
    x       = abs(x0(x0>1e-2/n)); 
    sx      = sort(x(x~=0));  
    [mx,it] = max(normalize(sx(2:end)./sx(1:end-1))) ;
    th      = 0; 
    if mx   > 10 && it(1)>1
       th   = sx(it(1)) ; 
    end         
    x0(abs(x0)<=th)=0; 
end