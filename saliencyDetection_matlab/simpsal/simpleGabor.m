function filter = simpleGabor(angle, phase)

if ( nargin == 1 )
  filter = {};
  filter.g0 = simpleGabor(angle,0);
  filter.g90 = simpleGabor(angle,90);
  return;
end

major_stddev = 2;
minor_stddev = 4;
max_stddev = 4;

sz = ceil(max_stddev*sqrt(10));

psi = pi / 180 * phase;
rtDeg = pi / 180 * angle;

omega = 2;
co = cos(rtDeg);
si = -sin(rtDeg);
major_sigq = 2 * major_stddev^2;
minor_sigq = 2 * minor_stddev^2;

vec = [-sz:sz];
vlen = length(vec);
vco = vec*co;
vsi = vec*si;

major = repmat(vco',1,vlen) + repmat(vsi,vlen,1);
major2 = major.^2;
minor = repmat(vsi',1,vlen) - repmat(vco,vlen,1);
minor2 = minor.^2;

result = cos(omega * major + psi) .* ...
         exp(-major2 / major_sigq ...
             -minor2 / minor_sigq);

% normalization
filter = result - mean(result(:));
filter = filter / sqrt(sum(filter(:).^2));
