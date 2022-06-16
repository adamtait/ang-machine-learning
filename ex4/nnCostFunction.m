function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

function Yvm = labelValueToVector(Ym)
  Yvm = zeros( size(Ym,1), size(Ym,2) );
  for i = 1:size(Ym,1)
    v = Ym(i);
    Yvm(i, v) = 1;
  endfor
end

function O = matrixMultiplyRowByRow(M, N)
  % assumes that M & N have same dimensions
  O = zeros(size(M,1), 1);

  for i = 1:size(M,1)
    v = 0
    for j = 1:size(M,2)
      v = v + M(i,j) * N(i,j)
    endfor
    O(i,1) = v;
  endfor
end


Theta0Vec = ones( size(X,1), 1 );

X1 = [ Theta0Vec X ];  % add Theta0. 5000x400 => 5000x401
P1 =  sigmoid( X1 * Theta1' );  % 5000x401 * 401x25  => 5000x25
P1 = [ Theta0Vec P1 ]; % 5000x25 => 5000x26
P = sigmoid( P1 * Theta2' ); % 5000x26 * 26x10  => 5000x10

Pn = 1 - P;

Yv = labelValueToVector(y) ; % 5000x1 => 5000x10
Yvn = 1 - Yv;


MEAN_TERM = m ^ -1;
SUM_TERM = -1 * ( ... % row-wise multiplication  [can also be implemented as matrixMultiplyRowByRow]
                 sum( log(P) .* Yv, 2 ) + ...
                 sum( log(Pn) .* Yvn, 2 ) ...  % 5000x10 , 5000x10 => 5000x1
                 );
J = MEAN_TERM * sum(SUM_TERM);


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end