function [out1,out2] = funcSimpleEx(x,key,T1,T2)
    % This code provides information for
    %     min   x'*[6 5;5 8]*x+[1 9]*x-sqrt(x'*x+1) 
    %     s.t. \|x\|_0<=s
    % where s=1   
    a   = sqrt(sum(x.*x)+1);
    switch key
        case 'fg'    
            out1 = x'*[6 5;5 8]*x+[1 9]*x-a;       % objective
            if  nargout == 2 
                out2 = 2*[6 5;5 8]*x+[1; 9]-x./a;  % gradient
            end
        case 'h'
            H   = 2*[6 5;5 8]+(x*x'-a*eye(2))/a^3; % sub-Hessian on (T1 T1) 
            out1 = H(T1,T1);
            if  nargout == 2 
                out2 = H(T1,T2);                   % sub-Hessian on (T1 T2) 
            end
    end
end
