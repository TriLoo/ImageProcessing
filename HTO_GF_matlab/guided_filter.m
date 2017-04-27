function filtered = guided_filter(input, guide, epsilon, win_size)
% calculate a few useful parameters
num_pixels = win_size * win_size;
half = floor(win_size / 2);
% average value in every win_size-by-win_size window of the input image
paddedp = padarray(input, [half, half], 'both');
mup = zeros(size(paddedp));
% average value in every win_size-by-win_size window of the guide image
paddedi = padarray(guide, [half, half], 'both');
mui = zeros(size(paddedi));
% variance in every window of the guide image
sigmai = zeros(size(paddedi));
% cross term of the guide and input image in every window
cross = zeros(size(paddedi));

%constructing denominator image;
initial_denom = padarray(ones(size(input)), [half, half], 'both');
denom = zeros(size(paddedi));


% calculating sum over each window by shifting and adding
for i = -half : half
    for j = -half : half
        mup = mup + circshift(paddedp, [i, j]);
        mui = mui + circshift(paddedi, [i, j]);
        sigmai = sigmai + circshift(paddedi, [i, j]).^2;
        cross = cross + circshift(paddedi, [i, j]).*circshift(paddedp, [i, j]);
        denom = denom + circshift(initial_denom, [i, j]);
    end
end

% remove the padding
mup = mup(half+1:end-half, half+1:end-half);
mui = mui(half+1:end-half, half+1:end-half);
sigmai = sigmai(half+1:end-half, half+1:end-half);
cross = cross(half+1:end-half, half+1:end-half);
denom = denom(half+1:end-half, half+1:end-half);

% calculate average, variance and cross terms in equation (5) and (6) in
% the paper
mup = mup ./ denom;
mui = mui ./ denom;
sigmai = sigmai ./ denom - mui.^2;
cross = cross ./ denom;

% calculating the linear coefficients a and b
a = (cross - mui .* mup) ./ (sigmai + epsilon);
b = mup - a .* mui;

apad = padarray(a, [half, half], 'both');
bpad = padarray(b, [half, half], 'both');

mua = zeros(size(apad));
mub = zeros(size(bpad));

% calculating sum over each window by shifting and adding
for i = -half : half
    for j = -half : half
        mua = mua + circshift(apad, [i, j]);
        mub = mub + circshift(bpad, [i, j]);
    end
end

% remove the padding
mua = mua(half+1:end-half, half+1:end-half);
mub = mub(half+1:end-half, half+1:end-half);

% calculate average a and b
mua = mua ./ denom;
mub = mub ./ denom;

% the filtered image
filtered = mua .* input + mub;
  
    
    










