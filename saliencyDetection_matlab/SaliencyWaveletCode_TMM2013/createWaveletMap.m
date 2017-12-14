function [ca,ch,cv,cd,ROW,COL] = createWaveletMap(img, waveletLevel,StrWaveMODE)

% this function can be used with 2D data

img = double(img);

img = mat2gray(img)*255;

[R,C] = size(img);

for i = 1:waveletLevel
    if i == 1
        [ca(i).data, ch(i).data, cv(i).data, cd(i).data] = dwt2(img,StrWaveMODE);
    else
        [ca(i).data, ch(i).data, cv(i).data, cd(i).data] = dwt2(ca(i-1).data,StrWaveMODE);
    end
    [ROW(i).data, COL(i).data] = size(ca(i).data);
    if(double(R)/2^i < 1 && double(C)/2^i < 1)
        fprintf('%d wavelet level processed!\n',i);
        break;
    end
end
