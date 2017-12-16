function R=PCNN_large_arrange3(matrix,max,link_arrange,np,pre_flag)

disp('PCNN is processing...')
[p,q]=size(matrix);

% computes the normalized matrix of the matrixA and  matrixB
F_NA=Normalized(matrix);
F_NB=Normalized(max);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize the parameters. 
% You'd better change them according to your applications 
alpha_L=1;
alpha_Theta=2.0;
alpha_F=0.1;
beta=0; 
 vF=0.5;
vL=0.2;
vTheta=20;
% Generate the null matrix that could be used
L1=zeros(p,q);
L2=zeros(p,q);
U1=zeros(p,q);
U2=zeros(p,q);
Y1=zeros(p,q);
Y2=zeros(p,q);
F1=Y1;
F2=Y1;
Y0=zeros(p,q);
Y01=zeros(p,q);
R=zeros(p,q);
Theta1=zeros(p,q);
Theta2=zeros(p,q);

% Compute the linking strength.
center_x=round(link_arrange/2);
center_y=round(link_arrange/2);
W=zeros(link_arrange,link_arrange);
for i=1:link_arrange
    for j=1:link_arrange
        if (i==center_x)&&(j==center_y)
            W(i,j)=0;
        else
            W(i,j)=1./sqrt((i-center_x).^2+(j-center_y).^2);
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%
I1=F_NA;
I2=F_NB;
for n=1:np
    [a,b]=size(W);
      for i=1:p
       for j=1:q
    for c=1:a
        for d=1:b
            
            
 
           beta=beta+[matrix(c,d)-matrix(c+1,d)].^2+[matrix(c,d)-matrix(c,d+1)].^2;
        end
    end
    K1(i,j)=conv2(Y1(i,j),W,'same');
    K2(i,j)=conv2(Y2(i,j),W,'same');
    F1(i,j)=exp(-alpha_F)*F1(i,j)+vF*K1(i,j)+I1(i,j);
    F2(i,j)=exp(-alpha_F)*F2(i,j)+vF*K2(i,j)+I2(i,j);
    L1(i,j)=exp(-alpha_L)*L1(i,j)+vL*K1(i,j);
    L2(i,j)=exp(-alpha_L)*L2(i,j)+vL*K2(i,j);
    Theta1(i,j)=exp(-alpha_Theta)*Theta1(i,j)+vTheta*Y1(i,j);
    Theta2(i,j)=exp(-alpha_Theta)*Theta2(i,j)+vTheta*Y2(i,j);
    U1(i,j)=F1(i,j).*(1+beta*L1(i,j));
    U2(i,j)=F2(i,j).*(1+beta*L2(i,j));
    Y1(i,j)=im2double(U1(i,j)>Theta1(i,j));
    Y2(i,j)=im2double(U2(i,j)>Theta2(i,j));
    Y0(i,j)=Y0(i,j)+Y1(i,j);
    Y01(i,j)=Y01(i,j)+Y2(i,j);
    end
      end
      Y0=medfilt2(Y0,[3,3]);
      Y01=medfilt2(Y01,[3,3]);
      for i=1:p
          for j=1:q
              
    if abs(Y0(i,j)-Y01(i,j))<=0.015
       R(i,j)=(I1(i,j)+I2(i,j))/2;
    end 
    if abs(Y0(i,j)-Y01(i,j))>0.015 &&Y0(i,j)>Y01(i,j)
        R(i,j)=I1(i,j);
    end
    if abs(Y0(i,j)-Y01(i,j))>0.015 && Y0(i,j)<Y01(i,j)
        R(i,j)=I2(i,j);
    end 
       end
      end
end


