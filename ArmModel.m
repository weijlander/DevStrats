% Author: Wouter Eijlander
% Provides a simulated arm in 3D to move in its world-space
classdef ArmModel
    %% Initialize arm variables and reachable space
    properties
        arm % vectors representing the endpoints of the upper arm and forearm respectively
        limits % default limits of joint rotation for each axis. NOTE: THESE MAY BE WRONG-WAY ROTATIONS
    end
    methods
        function obj = ArmModel(lengths, limits)
            if nargin == 0
                obj.arm = [0,0,-10;0,0,-20]; % vectors representing the endpoints of the upper arm and forearm respectively
                obj.limits = [-10,130; -10,100; -90,90; 0,140]; % default limits of joint rotation for each axis. NOTE: THESE MAY BE WRONG-WAY ROTATIONS
            else
                obj.arm = lengths;
                obj.limits = limits;
            end
        end
        function E = move(obj, muscles, coac)
            E = move_arm(obj.arm, muscles, obj.limits, coac);
        end
    end
end

