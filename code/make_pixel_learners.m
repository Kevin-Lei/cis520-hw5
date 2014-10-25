function [Yw] = make_pixel_learners(X)
% Converts pixel vectors into a pool of weak learners for AdaBoost.
% 
% Usage:
%
%    YW = make_pixel_learners(X)
%
% If X is a N x P matrix of N images with P pixels, returns a (-1,1) N x M 
% binary matrix of weak learners where M = 2P, representing the weak learner 
% associated with predictions from a single pixel. E.g., pixel #1 produces 
% weak learners #1 and #P+1, corresponding to predicting +1 when pixel 1 is
% >0 and predicting +1 when pixel 1 is ==0 respectively.

% Predict 1 if X > 0.
Yw = double(X>0);

% Predict -1 if X == 0.
Yw(Yw==0) = -1;

% Also add the opposite predictions to pool of weak learners.
Yw = [Yw -Yw];
