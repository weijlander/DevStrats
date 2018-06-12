% Calculates new end-effector position on a given 4 DOF arm. inputs:
% length vectors for the 3 dof upper arm and 1 dof forearm
% activations for the 8 muscles influencing these 3 dof: 
%   (8,1) vector of real values between 0 and 1
% Coactivation coefficient: 
%   real between 0 and 1
% Joint limits for each DOF:
%   (4,1) vector of degrees
function E = move_arm(arm, muscles,limits,coac)
% determine start positions for both arm segments
l1 = arm(1,:);
l2 = arm(2,:);

% determine rotation matrices for both arm segments
x1 = unify_muscles(muscles(1),muscles(2),coac);
y1 = unify_muscles(muscles(3),muscles(4),coac);
z1 = unify_muscles(muscles(5),muscles(6),coac);
x2 = unify_muscles(muscles(7),muscles(8),coac);

rx1 = limits(1,1)+(x1*(limits(1,2)-limits(1,1)));
ry1 = limits(2,1)+(y1*(limits(2,2)-limits(2,1)));
rz1 = limits(3,1)+(z1*(limits(3,2)-limits(3,1)));
rx2 = limits(4,1)+(x2*(limits(4,2)-limits(4,1)));

% determine positional relationships between segments, needed for correctly
% calculating forearm movement
d2 = l2-l1;

%% calculate end-effector position
e1 = homog_transform(l1(1),l1(2),l1(3),0,0,0,rx1,ry1,rz1); % Perform the rotation to the upper arm
e2 = homog_transform(d2(1),d2(2),d2(3),e1(1),e1(2),e1(3),0,0,rz1); % determine the new forearm position based on upper arm translation and rotation
e3 = homog_transform(e2(1),e2(2),e2(3),0,0,0,rx2,0,0); % perform forearm rotation
E = [e1, e3];