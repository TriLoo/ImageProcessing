function outData = Horizontal_Sinc_interpolation(inData)
% 
% This file implements: subpixel interpolation with popular Sinc
% interpolation technique.
% Has to interpolation three new subpixels between two integer index pixels
% in origin image. So the mySample is 4, i.e. upsample by 4
% 


Sinc = [  -0.0110    0.0452   -0.1437    0.8950    0.2777   -0.0812    0.0233   -0.0158;   
          -0.0105    0.0465   -0.1525    0.6165    0.6165   -0.1525    0.0465   -0.0105;
          -0.0053    0.0233   -0.0812    0.2777    0.8950   -0.1437    0.0452   -0.0110 ;]; 
mySample = 4;
[row,col] = size(inData);
outData = zeros(row,col*4);
sum1=0;sum2=0;sum3=0;
for i = 1 :row            
    for j = 1: col
        outData(i,j*4-3) = inData(i,j);
        for l=-mySample+1:mySample
           x=j+l;
           if x<1
               x=-x+2;
           end
           if x>col
               x=j*2-x;
           end
           sum1=sum1 + inData(i,x)*Sinc(1,l+mySample);
           sum2=sum2 + inData(i,x)*Sinc(2,l+mySample);
           sum3=sum3 + inData(i,x)*Sinc(3,l+mySample);
        end
        outData(i,j*4-2) = sum1;
        outData(i,j*4-1) = sum2;
        outData(i,j*4) = sum3;
        sum1=0;sum2=0;sum3=0;
    end
end


                   