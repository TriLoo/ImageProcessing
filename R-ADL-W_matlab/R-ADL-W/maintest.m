clc;
clear;
% origin  = imread('../datas/source20_1.tif');
origin = imread('lena.bmp');      % error, dimension dismatch.
[MyHeight,MyWidth] = size(origin);
%列变换
ori_QuartPixel_interp  = Horizontal_Sinc_interpolation(origin); %插值
[origin,Predict_direc_H] = Horizontal_Direction_Prediction(origin,ori_QuartPixel_interp);%预测

ori_QuartPixel_interp  = Horizontal_Sinc_interpolation(origin);%插值
origin = Horizontal_Update(origin,ori_QuartPixel_interp,Predict_direc_H);%更新

origin = origin';




%行变换
ori_QuartPixel_interp  = Horizontal_Sinc_interpolation(origin);    % subpixels interpolation
[origin,Predict_direc_V] = Horizontal_Direction_Prediction(origin,ori_QuartPixel_interp);    % predict

ori_QuartPixel_interp  = Horizontal_Sinc_interpolation(origin);    % subpixels interpolation
origin = Horizontal_Update(origin,ori_QuartPixel_interp,Predict_direc_V);   % update
origin = origin';

origin1 = abs(origin);
origin1 = uint8(origin1);
figure(2)
subplot(121),imshow(origin1);









%行逆变换
origin = origin';

ori_QuartPixel_interp  = Horizontal_Sinc_interpolation(origin);%插值
origin = inverse_Horizontal_Update(origin,ori_QuartPixel_interp,Predict_direc_V);%逆更新
ori_QuartPixel_interp  = Horizontal_Sinc_interpolation(origin);%插值
origin = inverse_Horizontal_Prediction(origin,ori_QuartPixel_interp,Predict_direc_V);%逆预测
%列逆变换
origin = origin';

ori_QuartPixel_interp  = Horizontal_Sinc_interpolation(origin);
origin = inverse_Horizontal_Update(origin,ori_QuartPixel_interp,Predict_direc_H);
ori_QuartPixel_interp  = Horizontal_Sinc_interpolation(origin);
origin = inverse_Horizontal_Prediction(origin,ori_QuartPixel_interp,Predict_direc_H);

origin1 = abs(origin);
origin1 = uint8(origin1);
subplot(122),imshow(origin1);


