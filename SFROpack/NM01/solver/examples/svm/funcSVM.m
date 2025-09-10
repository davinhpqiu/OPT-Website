function out = funcSVM(x,key,w,A,c)
    n = length(x);
    switch key   
        case 'f';  out = norm(x,'fro')^2 - (1-w)*x(n)^2;
        case 'g';  out = x;  out(n) = w*x(n);
        case 'h';  out = speye(n); out(n,n) = w; 
        case 'a';  acc = @(var)nnz( sign( (A*var(1:n-1)+var(n)) )-c);
                   out = 1-acc(x)/length(c);
        otherwise; out = []; % 'Otherwise' is REQIURED
    end    
end