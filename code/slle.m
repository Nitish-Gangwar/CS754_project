%% slle function for spherical Locally Linear Embedding
% Reference: https://cs.nyu.edu/~roweis/lle/code.html
function [Y,Z] = slle(X,K,d)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% size of X is D*N where N is number of points D is the dimensionality. 
    [D,N] = size(X);
    
    
    fprintf(1,'sLLE running on %d points in %d dimensions\n',N,D);


    %% STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS 
    fprintf(1,'-->Finding %d nearest neighbours.\n',K);
    
    
    %%% computing distances between everypair 
    distance = zeros(N);
    for i=1:N
        for j=1:N
            distance(i,j) = norm(X(:,i) -X(:,j));
        end
    end
    
    disp("size of distance matrix is")
    disp(size(distance))
    
    
    %%% sorting distance
    [~,index] = sort(distance);
    
    %%% neighbourhood array contains indices of first k neighbours
    neighborhood = index(2:(1+K),:);



    %% STEP2: SOLVE FOR RECONSTRUCTION WEIGHTS
    fprintf(1,'-->Solving for reconstruction weights.\n');

    if(K>D) 
      fprintf(1,'   [note: K>D; regularization will be used]\n'); 
      tol=1e-3; % regularlizer in case constrained fits are ill conditioned
    else
      tol=0;
    end
    
    %%% W contains the weight vectors
    W = zeros(K,N);
    for ii=1:N
       z = X(:,neighborhood(:,ii))-repmat(X(:,ii),1,K); % shift ith pt to origin
       C = z'*z;                                        % local covariance
       C = C + eye(K,K)*tol*trace(C);                   % regularization (K>D)
       W(:,ii) = C\ones(K,1);                           % solve Cw=1
       W(:,ii) = W(:,ii)/sum(W(:,ii));                  % enforce sum(w)=1
    end


    %% STEP 3: COMPUTE EMBEDDING FROM EIGENVECTS OF COST MATRIX M=(I-W)'(I-W)
    fprintf(1,'-->Computing embedding.\n');

    M=eye(N,N); % use a sparse matrix with storage for 4KN nonzero elements
    %     M = sparse(1:N,1:N,ones(1,N),N,N,4*K*N); 
    for ii=1:N
       w = W(:,ii);
       jj = neighborhood(:,ii);
       M(ii,jj) = M(ii,jj) - w';
       M(jj,ii) = M(jj,ii) - w;
       M(jj,jj) = M(jj,jj) + w*w';
    end

    % CALCULATION OF EMBEDDING 
    B_old = zeros(N);
    Y_old = zeros(d,N);
    Y = ones(d,N);
    B = eye(N);
    while norm(Y-Y_old)>=1e-7 && norm(B-B_old)>=1e-7
%     for i=1:500
        Y_old = Y';
        [Y_temp,Diag,~] = eig(M,B);
    %     [Y_temp,Diag] = eigs(M,B,d,'sm', options);
        [~, b]=sort(diag(Diag));
        Y = Y_temp(:,b(1:d));
    %     Y = Y_temp(1:2,:)';
    %     B = diag(diag(inv((Y)*(Y'))));
        B_old = B;
        for j=1:N
            B(j,j) = 1/(Y_temp(j,:)*Y_temp(j,:)');
        end
%         disp(size(Y));
%         disp(size(Y_old));
%         disp(size(B));
%         disp(size(B_old));
    end

    Z = real(sqrtm(B)*Y);

    fprintf(1,'Done.\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


