function [out1,out2] = funcLinReg(x,key,T1,T2,A,b)
    % This code provides information for
    %     min   0.5*||Ax-b||^2 
    %     s.t. \|x\|_0<=s
    % where A in R^{m x n} and b in R^{m x 1}    
    switch key
        case 'fg'
            Tx   = find(x~=0);
            Axb  = A(:,Tx)*x(Tx)-b;
            out1 = (Axb'*Axb)/2;      % objective 
            if  nargout == 2 
                out2 = (Axb'*A)';     % gradient 
            end
        case 'h'        
            AT   = A(:,T1); 
            out1 = AT'*AT;            %sub-Hessian formed by rows indexed by T1 and columns indexed by T1   
            if  nargout == 2
                out2 = AT'*A(:,T2);   %sub-Hessian formed by rows indexed by T1 and columns indexed by T2
            end       
    end
end




