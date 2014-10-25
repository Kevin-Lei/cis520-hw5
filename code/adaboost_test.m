function [test_err margins] = adaboost_test(boost, Yw_test, Ytest)
% Generates predictions for AdaBoost on new data.
%
% Usage:
%
%   TEST_ERR MARGINS = adaboost_test(BOOST, YW_TEST, YTEST)
%
% Returns the predictions by Adaboost given a weighted combination of weak
% learners stored in the struct BOOST. YW is the predictions of the same
% pool of weak learners for the new data.

% Compute test error and margin
Yhat = zeros(size(Ytest, 1), 1);
for t = 1:numel(boost.h)
    Yhat = Yhat + (boost.alpha(t) * Yw_test(:,boost.h(t)));
    
    test_err(t) = sum((Yhat .* Ytest)<0) / size(Ytest,1);
    margins{t} = (Yhat .* Ytest) / sum(abs(boost.alpha(1:t)));
end
