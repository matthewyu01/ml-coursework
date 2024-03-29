function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, length(theta));
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    hypothesis = X*theta

    theta = theta - (alpha / m) * X' * (hypothesis-y)


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter,1:length(theta)) = computeCost(X, y, theta);

end

end
