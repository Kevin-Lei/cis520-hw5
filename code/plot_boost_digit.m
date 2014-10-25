function [h] = plot_boost_digit(boost, x, T)
% PLOT_DT_DIGIT - Plots the pixels used by AdaBoost.
%
% Usage:
%
%   H = PLOT_DT_DIGIT(boost, X [, T])
%
%  Returns the handle to an image plot. The image shows the handdrawn digit
%  in red, with the pixels chosen by AdaBoost to evaluate the image
%  highlighted, colored by the order in which they were evaluated (bright
%  green = first, dark blue = last). Optionally shows only the first T
%  rounds.

if nargin==2
    T = numel(boost.h)
end
fidx = boost.h(1:T);
is_neg = fidx > size(x,2);
fidx(fidx > size(x,2)) = fidx(fidx > size(x,2)) - size(x,2);


% Convert the input feature vector into an RGB image, with the red channel
% greater than the others.
im = reshape(x, [28 28])';
im_rgb = repmat(im, [1 1 3]);

% Generate an image with just the pixels that were chosen by AdaBoost, so
% that they show up in the correct spot in the visualization.
x_f = zeros(size(x));
x_f(fidx) = 1:numel(fidx);
im_f = reshape(x_f, [28 28])';

% Find the row and col indices of the selected pixels.
[i j] = find(im_f);

% Prepare colors for the pixels (using scheme 'winter').
colors = repmat([255 0 0], numel(fidx), 1); %wint(numel(fidx)).*255;
%colors = colors(end:-1:1, :);

% Color the pixels in the letter image.
for k = 1:numel(i)
    if is_neg(im_f(i(k), j(k)))
        im_rgb(i(k),j(k),:) = reshape(colors(im_f(i(k),j(k)),:), 1, 1, 3);
    else
        im_rgb(i(k),j(k),:) = reshape(colors(im_f(i(k),j(k)),end:-1:1), 1, 1, 3);
    end
end

% Display image for the user with appropriate title.
h = image(uint8(im_rgb)); axis image; axis off;
%title(['P(Y) = ' num2str(y, '%.2f,')]);