function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   a2_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, a2_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:a2_layer_size * (input_layer_size + 1)), ...
                 a2_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (a2_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (a2_layer_size + 1));

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

% layer unit count: 400， 25, 10
% X, 5000, 400
% Theta1 25 , 401
% Theta2 10, 26
% y 5000, 1
Y = zeros(num_labels, m);
for i = 1:m
    Y(y(i, 1), i) = 1;
end

%size(Theta1)
%size(Theta2)
%size(X)
a1 = X;%5000 x 400
a1_bis = [ones(m, 1), X]';%401 x 5000 
z2 = Theta1 * (a1_bis);% 25 x 5000
a2 = sigmoid(z2);% 25 x 5000
a2_bis = [ones(m, 1), a2'];%26 x 5000
z3 = Theta2 * (a2_bis');%10 x 5000
a3 = sigmoid(z3);%10 x 5000
%size(Y.*log(h) - (1 - Y).*log(h))
%size(a3)
size(a3)
J = (1 / m) * sum(sum( -Y.*log(a3) - (1 - Y).*log(1- a3) ))

%Regularized
sumTheta1 = 0.0;
for i = 1:size(Theta1, 1)
    for j = 2:(size(Theta1, 2))
        sumTheta1 = sumTheta1 + Theta1(i, j)*Theta1(i, j);
    end
end

sumTheta2 = 0.0;
for i = 1:size(Theta2, 1)
    for j = 2:(size(Theta2, 2))
        sumTheta2 = sumTheta2 + Theta2(i, j)*Theta2(i, j);
    end
end
J = J + (lambda/(2 * m)) * (sumTheta1 + sumTheta2)

% -------------------------------------------------------------
bigDelta2 = zeros(num_labels, a2_layer_size);
bigDelta1 = zeros(a2_layer_size, input_layer_size);
for i = 1:m
    if mod(i, 1000) == 0,
        fprintf('iter: %d\n', i);
    end
    item_a3 = a3(1:num_labels, i); % 10 x 1
    item_Y = Y(1:num_labels, i); % 10 x 1
    delta3 = item_a3 - item_Y; % 10 x 1
    
    item_z2 = z2(1:a2_layer_size, i);%25 x 1
    delta2 = (Theta2(:, 2:end)' * delta3) .* sigmoidGradient(item_z2);%25 * 1
    
    item_a2 = a2(1:a2_layer_size, i);%25 x 1
    bigDelta2 = bigDelta2 + delta3 * (item_a2');%10 x 25
    
    item_a1 = a1'(1:input_layer_size, i);% 400 x 1
    bigDelta1 = bigDelta1 + delta2 * (item_a1');%25 x 400
    
end
Theta1_grad = (1 / m) * [ones(a2_layer_size, 1), bigDelta1];
Theta2_grad = (1 / m) * [ones(num_labels, 1), bigDelta2];
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
