function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

MEAN_TERM = m ^ -1;

yi = 1 - y;  % y inverse
P = sigmoid( X * theta );  % Prediction
Pi = 1 - P;  % P inverse

SUM_TERM = (log(P)' * y) + (log(Pi)' * yi);
REG_TERM = MEAN_TERM * lambda * (2 ^ -1) * sum( theta(2:end) .^ 2 );

J = (MEAN_TERM * -1 * SUM_TERM) + REG_TERM;


% gradient

ERROR = P - y;
GRAD_REG_TERM = [0; (lambda / m) * theta(2:end)];   % 28x1
grad = ( ( m ^ -1 ) * ( ERROR' * X )' ) + GRAD_REG_TERM; % 1x28 + 1x28

% =============================================================

end
