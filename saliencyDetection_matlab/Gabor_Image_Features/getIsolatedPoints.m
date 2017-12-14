function i=getIsolatedPoints(dBinary)
%
% Returns isolated points from a binarized version of the
% pairwise distance matrix (1, if distance is less than a
% threshold, and 0 otherwise)
%
i=find(sum(dBinary)==1);    % those whose distance is less than
                            % a threshold only at the diagonal