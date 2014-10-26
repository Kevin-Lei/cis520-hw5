%% Written / report part of Question 1 for Homework 6

%% Put your written answers here.
clear all
answers{1} = 'There is very little overfitting, because the test error is only slightly higher than that of the training error. The training error does not seem to be going towards zero.';
answers{2} = 'The margin is getting smaller with larger T. This is because, the more iterations that we use adaboost on, the more weak classifiers we have to help us classify better. However, the margin will never go towards zero, because the data is not perfectly separable.';
answers{3} = 'Adaboost is getting the images on the left wrong because the location the numbers were written were in are not where the weak classifers are. Increasing T would not help that much because the numbers can still be in a location such that it is hard to predict correctly. Instead, instead of picking 1 pixel, each weak classifier should pick 2x2 pixels to make it more accurate.';

save('problem_answers.mat', 'answers');

%% Follow the instructions below to generate plots.

data = load('../data/mnist_all.mat');

[X Y] = get_digit_dataset(data, {'3', '5'}, 'train');
[Xtest Ytest] = get_digit_dataset(data, {'3', '5'}, 'test');

% Create weak learner pool for training and test data.
Yw = make_pixel_learners(X);
Yw_test = make_pixel_learners(Xtest);

%% Train AdaBoost for 200 rounds, and evaluate on the test data.
T = 200;

% Run adaboost for 200 arounds
boost = adaboost_train(Y, Yw, T);

% Compute test error and margins
[test_err margins] = adaboost_test(boost, Yw_test, Ytest);

%% Plot 1 - Training err, Test err vs. T

figure('Name', 'Train vs. Test Err of Adaboost');
hold on;
plot(1:T, boost.train_err, '--r', 'LineWidth', 2);
plot(1:T, test_err, '-b', 'LineWidth', 2);
hold off;
legend({'Train Err', 'Test Err'});
xlabel('T');
ylabel('Error');
title('Train err vs. Test error of Adaboost');

% Save plot to disk
print -djpeg -r72 plot_1.jpg

%% Plot 2 - Margin vs T
clear h;

trange = [5 25 200];
figure('Name', 'Margin distribution vs. T');
hold on;
cols = {'b','r','g'};
for i = 1:3
    yhat = margins{trange(i)};
    h(i) = cdfplot(yhat);
    
    set(h(i), 'LineWidth', 2);
    set(h(i), 'Color', cols{i});
    line([mean(yhat) mean(yhat)], [0 1], 'LineStyle', '--', 'Color', cols{i});
end
hold off;

% Make graph nicely labelled and legend'ed
xlabel('Margin on Test Data');
ylabel('P(Margin >= x)');
xlim([-1 1]);
title('CDF of Margin on Test Data');
legend(h, arrayfun(@(x)sprintf('T=%d', x), trange, 'UniformOutput', false));

% Save plot to disk
print -djpeg -r72 plot_2.jpg

%% Plot 3 - Understanding Mistakes
figure('Name', 'Worst vs. Best Test Example');

idx5 = find(Ytest==1);
idx1 = find(Ytest==-1);

% Look at all examples of 3's, and find the examples with lowest and
% highest margin.
[worst_margins, sort_order] = sort(margins{end}(idx1));

subplot(2,2,1);
i = 1;
plot_boost_digit(boost, Xtest(idx1(sort_order(i)),:), 50);
title(sprintf('margin = %.2f', worst_margins(i)));

subplot(2,2,2);
i = numel(idx1);
plot_boost_digit(boost, Xtest(idx1(sort_order(i)),:), 50);
title(sprintf('margin = %.2f', worst_margins(i)));

% Now do the same, but only looking at examples of 5's.
[worst_margins, sort_order] = sort(margins{end}(idx5));

subplot(2,2,3);
i = 1;
plot_boost_digit(boost, Xtest(idx5(sort_order(i)),:), 50);
title(sprintf('margin = %.2f', worst_margins(i)));

subplot(2,2,4);
i = numel(idx5);
plot_boost_digit(boost, Xtest(idx5(sort_order(i)),:), 50);
title(sprintf('margin = %.2f', worst_margins(i)));

print -djpeg -r72 plot_3.jpg

