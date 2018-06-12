function [x,y] = find_coord(angle_upper, angle_lower, data1, data2)
% finds the end x and y position from angles given and extracted from data1
% and data2
    indicesUp = find(data1(:,3)==angle_upper)  % somehow is fine?
    indicesLow = find(data2(:,3)==angle_lower) % does not work somehow
    index = intersect(indicesUp,indicesLow);
    v = data1(index,1:2);   % <- black magic
    x = v(1);
    y = v(2);
end


