function outStr=point2p(num)

s=num2str(num,"%.2f");

outStr=strrep(s,'.','p');
