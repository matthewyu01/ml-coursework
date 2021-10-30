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


hypothesis = sigmoid(X * theta);

summed_error = 1/m * sum(-1*y .* log(hypothesis) - (1-y) .* log(1-hypothesis)); 

summed_penalty = lambda / (2*m) * (sum(theta.^2) - theta(1).^2);

J = summed_error + summed_penalty;


grad = ( 1/m * sum((hypothesis-y).*X) + (lambda/m * theta )'); %tranposed to add 3x1 with 1x3

grad(1) = 1/m * sum((hypothesis-y).*X(1)); %j_0 theta

% =============================================================

end
