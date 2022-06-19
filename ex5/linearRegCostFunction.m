function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Xb = [ones(m, 1) X];
% lambda = 1;


MEAN_TERM = 1 / m;

P = X * theta; % 12x2 * 2x1 => 12x1
ERROR_TERM = ( P - y ); % 12x1 - 12x1
SUM_TERM = sum( ERROR_TERM .^ 2 );
REG_TERM = lambda * sum( theta(2:end,:) .^ 2 );

J = ( 1/2 ) * MEAN_TERM * ( SUM_TERM + REG_TERM );


% gradient

%GRAD_ERROR_TERM = [ zeros(m,1) ( P - y ) ]; % 12x2
GRAD_SUM_TERM = ERROR_TERM' * X; % 1x12 * 12x2  => 1x2
GRAD_REG_TERM = lambda * [ zeros(1,size(theta,2)); theta(2:end,:) ]; % 2x1

grad = MEAN_TERM * ( GRAD_SUM_TERM' + GRAD_REG_TERM ); % 1x2 + 2x1 => 2x1

% =========================================================================

grad = grad(:);

end
