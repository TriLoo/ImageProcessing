function CM = createFeatureMap(ch, cv, cd, ROW, COL, R, C, waveletLevel, StrWaveMODE)

% this function can be used with 2D data
  
if waveletLevel > 1   
    for i = waveletLevel:-1:1
        if i == waveletLevel
            CM = idwt2([],ch(i).data, cv(i).data, cd(i).data,StrWaveMODE,[ROW(i-1).data COL(i-1).data]);
        elseif i == 1
%             CM = imresize(CM,[R C]);
%             continue;
            CM = idwt2(CM,ch(i).data, cv(i).data, cd(i).data,StrWaveMODE,[R C]);
%             CM = idwt2(CM,[], [], [],StrWaveMODE,[R C]);
        else
            CM = idwt2(CM,ch(i).data, cv(i).data, cd(i).data,StrWaveMODE,[ROW(i-1).data COL(i-1).data]);
        end
    end
elseif waveletLevel == 1
    CM = idwt2([],ch(1).data, cv(1).data, cd(1).data,StrWaveMODE,[R C]);
end

CM = (CM).^2/10000;





