function H = homog_transform(x,y,z,tx,ty,tz,dx,dy,dz)

T = eye(4);
T(1:3,4) = [tx,ty,tz];

Rx = [1 0 0 0; 0 cosd(dx) -sind(dx) 0; 0 sind(dx) cosd(dx) 0; 0 0 0 1] ;
Ry = [cosd(dy) 0 sind(dy) 0; 0 1 0 0; -sind(dy) 0  cosd(dy) 0; 0 0 0 1] ;
Rz = [cosd(dz) -sind(dz) 0 0; sind(dz) cosd(dz) 0 0; 0 0 1 0; 0 0 0 1] ;

xyz  = [x;y;z;1];

H = T*Rx*Ry*Rz*xyz;