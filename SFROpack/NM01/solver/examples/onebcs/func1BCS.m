function out = func1BCS(x,key,eps,q,A,c) 
    switch key   
        case 'f';  out = sum((x.^2+eps).^(q/2));
        case 'g';  out = q*x.*(x.^2+eps).^(q/2-1); 
        case 'h';  x2  = x.*x;
                   out = diag(( (x2+eps).^(q/2-2) ).*((q-1)*x2+eps) ); 
        case 'a';  acc = @(var)nnz(sign(A*var)-c);
                   out = 1-acc(x)/length(c);
        otherwise; out = []; % 'Otherwise' is REQIURED
    end    
end
