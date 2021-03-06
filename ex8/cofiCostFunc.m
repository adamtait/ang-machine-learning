function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% You need to return the following values correctly
J = 0;
X_grad = []; % zeros(size(X));
Theta_grad = []; %zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the
%                     partial derivatives w.r.t. to each element of Theta
%

J1 = X * Theta'; %  5x3 * 3x4 => 5x4
J2 = J1 - Y; % 5x4 - 5x4
J3 = R .* J2; % 5x4 .* 5x4
J4 = J3 .^ 2; % 5x4
J5 = ( 1 / 2 ) * J4; % 5x4

RegTerm = lambda * ( 2 ^ -1 );
RegT = RegTerm * sum( sum( Theta .^ 2 ) );
RegX = RegTerm * sum( sum( X .^ 2 ) );

J = sum(sum( J5 )) + RegT + RegX;


% gradient for X
for i = 1:num_movies
  Xgnwr = J3(i,:) * Theta; % 1x4 * 4x3 => 1x3
  Xgnr = lambda * X(i,:); % 1x3
  Xgn = Xgnwr + Xgnr; % 1x3 + 1x3 => 1x3
  X_grad = [ X_grad ; Xgn ];
endfor

% gradient for Theta
for i = 1:num_users
  Tgnwr = J3(:,i)' * X; % 1x5 * 5x3 => 1x3
  Tgnr = lambda * Theta(i,:); % 1x3
  Tgn = Tgnwr + Tgnr; % 1x3 + 1x3 => 1x3
  Theta_grad = [ Theta_grad ; Tgn ];
endfor



% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
