function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
tmpC = 0;
tmpsigma = 0;
error = 1;
for i=1:8
    for j = 1:8
        tmpC = Cs(i);
        tmpsigma = sigmas(j);

        x1 = [1 2 1]; x2 = [0 4 -1];
        model= svmTrain(X, y, tmpC, @(x1, x2) gaussianKernel(x1, x2, tmpsigma));
        predictions = svmPredict(model, Xval);
        tmperror = mean(double(predictions ~= yval));
        if(tmperror < error)
            error = tmperror;
            C = tmpC;
            sigma=tmpsigma;
        end

    end
end





% =========================================================================

end
