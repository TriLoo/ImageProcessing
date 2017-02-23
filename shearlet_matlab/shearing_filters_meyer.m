function [ w_s ] = shearing_filters_meyer( n1, level )
% -----------------------------------
% Author : smh
% Data   : 2017. 02. 23
% Description :
%       this function computes the directional/shearing filters using the
%       meyer window.
% -----------------------------------

% Inputs: n1 indicates the supports size of the directional filter is n1xn1
%         level indicates that the number of directions is 2^level 
%         
% Output: a sequence of 2D directional/shearing filters w_s where the
%         third index determines which directional filter is used

% generate indexing coordinate for Pseudo-Polar Grid
[x11,y11,x12,y12,F1]=gen_x_y_cordinates(n1);


wf=windowing(ones(2*n1,1),2^level);
w_s=zeros(n1,n1,2^level); %initialize window array
for k=1:2^level,
    temp=wf(:,k)*ones(n1,1)';
    w_s(:,:,k)=rec_from_pol(temp,n1,x11,y11,x12,y12,F1); % convert window array into Cartesian coord.
    w_s(:,:,k)=real(fftshift(ifft2(fftshift(w_s(:,:,k)))))./sqrt(n1);     %% why ? first fft, then ifft, and fftshift again ?
end

end

