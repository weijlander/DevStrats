classdef EyeModel
    % The eye model class, provides functionality for fixation and visual
    % input gathering
    
    properties
        default % vector with 3 values [x,y,z] encoding eye fixation at base activation.
        limits  % 3x2 vector of axis roation limits
        vnoise  % visual noise coefficient TODO MAKE THIS SOMETHING USABLE
        dir     % current direction of the eye, which can differ from the default direction
        fovd    % off-center degrees of the eye-centered field-of view 
        drange   % the range of the world-space
    end
    
    methods
         function obj = EyeModel(default, limits, vnoise, fovd, drange)
            if nargin == 0
                obj.default = [0,5,0]; % vector representing the direction of the eyes
                obj.limits = [-45,45;0,50;-45,45]; % default limits of joint rotation for each axis. NOTE: THESE MAY BE WRONG-WAY ROTATIONS
                obj.vnoise = 1;
                obj.dir = obj.default;
                obj.fovd = 50;
                obj.drange = -30:.1:30;
            else
                obj.default = default;
                obj.limits = limits;
                obj.vnoise = vnoise;
                obj.fovd = fovd;
                obj.drange = drange;
            end
         end
        
        function dir = move(obj,muscles,coac)
            % moves the model's gaze direction, and returns it for
            % convenience
            obj.dir = move_eyes(obj.default,muscles,obj.limits,coac);
            dir = obj.dir;
        end
        
        function ker = makeKernel(obj,dir)
            % returns the x,y and z-axis pdfs of the current fixation
            % point
            len = norm(dir);
            wid = len*(sind(obj.fovd)/cosd(obj.fovd));
            k=[];
            for d=1:1:3
                j = makedist('Normal',dir(d),wid/3);
                k = [k ; pdf(j,obj.drange)];
            end
            ker = k;
        end
        
        function vision = process_input(obj,dir,space)            
            % Get the x,y,z-axis values for the target, hand, and the
            % distributions for the visual kernel
            ker = obj.makeKernel(dir);
            st = space(1:3,:);
            sh = space(4:6,:);
            
            % Get the overlap between the objects and visual fixation kernel
            t = [];
            h = [];
            for d=1:1:3
                t = [t ; st(d,:).*ker(d,:)];
                h = [h ; sh(d,:).*ker(d,:)];
            end
            vision = [t;h];
            
        end
        
        % TODO:
        %   - meshgrid (?) to collapse the 6x601 into a 3d (601x601x601?)
        %   representation of locations
        %   - function to look around n times and minimize world-space
        %   uncertainty. Maybe not in this class though?
    end
    
end

