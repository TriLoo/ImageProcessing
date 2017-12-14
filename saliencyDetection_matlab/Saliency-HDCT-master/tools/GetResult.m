function [alpha TestDic] = GetResult(im, imlab, Sp, N_Sp, Locpixels, fore, back, Numbers)

    Numbers2 = zeros(length(fore)+length(back), length(fore)+length(back));
    for i = 1 : length(fore)+length(back)
       Numbers2(i,i) = Numbers(i);
    end

    imhsv = rgb2hsv(im);
    imlab(:,:,1) = mat2gray(imlab(:,:,1));  imlab(:,:,2) = mat2gray(imlab(:,:,2));  imlab(:,:,3) = mat2gray(imlab(:,:,3));
    [Dic] = GenerateDictionarySp(N_Sp, Sp, Locpixels, im, imhsv, imlab);
    TestDic = [];
    TestDic = [Dic.^0.5 Dic Dic.^1.5 Dic.^2.0];
    TestDic2 = [TestDic(fore,:); TestDic(back,:)];
    lengthdic = size(TestDic,2);

    %% Nonlinear Least-squares problem for 
    TrX = [ones(length(fore),1); zeros(length(back),1)];
    f = @(x)(sqrt(Numbers2)*(TestDic2*x - TrX));            
    options.Display = 'off';
    options.TolFun = size(im,1)*size(im,2)*10e-4;
    alpha =  lsqnonlin(f,zeros(lengthdic,1), ones(lengthdic,1).*-100, ones(lengthdic,1).*100, options);
