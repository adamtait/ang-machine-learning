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

%function O = matrixMultiplyRowByRow(M, N)
%  % assumes that M & N have same dimensions
%  O = zeros(size(M,1), 1);
%
%  for i = 1:size(M,1)
%    v = 0
%    for j = 1:size(M,2)
%      v = v + M(i,j) * N(i,j)
%    endfor
%    O(i,1) = v;
%  endfor
%end

function S = regularizationSum(T)
  S = sum(sum( T(:,2:end) .^ 2 ))
end



Theta0Vec = ones( size(X,1), 1 );

X1 = [ Theta0Vec X ];  % add Theta0. 5000x400 => 5000x401
P1 =  sigmoid( X1 * Theta1' );  % 5000x401 * 401x25  => 5000x25
P1 = [ Theta0Vec P1 ]; % add Theta0. 5000x25 => 5000x26
P = sigmoid( P1 * Theta2' ); % 5000x26 * 26x10  => 5000x10

Pn = 1 - P;

Yv = labelValueToVector(y) ; % 5000x1 => 5000x10
Yvn = 1 - Yv;


MEAN_TERM = m ^ -1;
SUM_TERM = -1 * ( ... % row-wise multiplication  [can also be implemented as matrixMultiplyRowByRow]
                 sum( log(P) .* Yv, 2 ) + ...
                 sum( log(Pn) .* Yvn, 2 ) ...  % 5000x10 , 5000x10 => 5000x1
                 );
REG_TERM_SUMS = regularizationSum(Theta1) + regularizationSum(Theta2);
REG_TERM = lambda * ( 2 ^ -1 ) * MEAN_TERM * REG_TERM_SUMS;

J = (MEAN_TERM * sum(SUM_TERM)) + REG_TERM;


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

for i = 1:5
  fprintf("%d", i);
endfor


for t = 1:m
  a_1 = [ 1 X(t,:) ]; % 1x401
  z_2 = a_1 * Theta1'; % 1x401 * 401x25 => 1x25
  a_2 = [ 1 sigmoid( z_2 ) ]; % 1x26
  z_3 = a_2 * Theta2'; % 1x26 * 26x10 => 1x10
  a_3 = sigmoid( z_3 ); % 1x10

  Yt = Yv(t,:); % 1x10
  d_3 = a_3 - Yt; % 1x10 - 1x10
  Theta2_grad = Theta2_grad + ( d_3' * a_2 ); % 10x26 + ( 10x1 * 1x26 )

  d_2 = ( Theta2(:,2:end)' * d_3' ) ... % 25x10 * 10x1 => 25x1
        .* sigmoidGradient( z_2' );     % 25x1 .* 25x1
  Theta1_grad = Theta1_grad + ( d_2 * a_1 ); % 25x401 + ( 25x1 * 1x401 )

endfor

Theta2_grad = MEAN_TERM * Theta2_grad;
Theta1_grad = MEAN_TERM * Theta1_grad;



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


function S = gradientRegularizationSum(T)
  S = MEAN_TERM * lambda * T(:,2:end);
  S = [ zeros(size(T,1),1) S ];
end

%fprintf("size(Theta2): %d %d\n", size(Theta2, 1), size(Theta2, 2));
%fprintf("grs: %f", gradientRegularizationSum(Theta2) );

Theta2_grad = Theta2_grad + gradientRegularizationSum(Theta2); % 3x6 + 3x5
Theta1_grad = Theta1_grad + gradientRegularizationSum(Theta1);





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
