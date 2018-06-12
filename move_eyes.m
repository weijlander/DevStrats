function dir = move_eyes(vis,muscles,limits,coac)
% turns the eye around the x- and z-axes (upward and sideward respectively)
agx = max(muscles(1:2));
antx = min(muscles(1:2));
agz = max(muscles(3:4));
antz = min(muscles(3:4));

% x and z are treated as antagonistic muscles- y is trated as a
% higher-level depth fixation value
x = unify_muscles(agx,antx,coac);
y = muscles(5);
z = unify_muscles(agz,antz,coac);

% x and z movement are rotations along axes, y movement is depth fixation
% and is thus a translation across the y axis (forward-backward)
rx = limits(1,1)+(x*(limits(1,2)-limits(1,1))); 
ty = limits(2,1)+(y*(limits(2,2)-limits(2,1)));
rz = limits(3,1)+(z*(limits(3,2)-limits(3,1)));

dir = homog_transform(vis(1),vis(2),vis(3),0,ty,0,rx,0,rz);