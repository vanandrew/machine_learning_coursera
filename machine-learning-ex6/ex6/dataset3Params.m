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

% Potential C and sigma values
p_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
p_sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% Initiallize this to a high value to minimize prediction error
min_prediction_error = 999999;

% Find C and sigma that minimizes the prediction error
for i=1:8
    for n=1:8
        model= svmTrain(X,y,p_C(i),...
            @(x1, x2) gaussianKernel(x1,x2,p_sigma(n)));
        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions~=yval));
        if prediction_error < min_prediction_error
            min_prediction_error = prediction_error;
            C = p_C(i);
            sigma = p_sigma(n);
        end
    end
end

% =========================================================================

end
