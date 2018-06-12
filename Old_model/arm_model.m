% Authors: Wouter Eijlander, Anouk van Maris, Luc Wijnen
% creates, trains and tests a neural network to research von Hofsten's
% findings in his paper "developmental changes in the organization of
% prereaching behaviour".
% used in combination with two scripts provided by Beata Grzyb:
% input_gauss_activation.m and sigmoid.m

l1 = 10;                % length of upper arm
l2 = 10;                % length of forearm
reach_thresh = 14;      % reaching distance threshold
reward = 0;
nexps = 2;

results_exReaches = zeros(nexps,7);
results_fixReaches = zeros(nexps,7);
results_unfixReaches = zeros(nexps,7);
results_derivFix = zeros(nexps,6);
results_derivMusc = zeros(nexps,6);

week_interval = [1 4 6 7 10 13 16 19];      % the week intervals from the von Hofsten paper. week 6 is present to change muscle coordination

for experiment=1:1:nexps
    extended_reaches = zeros(1,(length(week_interval)-1)); % vector in which the fixated reaches data is stored
    fixated_reaches = zeros(1,(length(week_interval)-1)); % vector in which the fixated reaches data is stored
    unfixated_reaches = zeros(1,(length(week_interval)-1)); % vector in which the unfixated reaches data is stored
    dist_errors = zeros(1,(length(week_interval)-1));
    fix_errors = zeros(1,(length(week_interval)-1));
    muscle_errors = zeros(1,(length(week_interval)-1));
    epsilons = zeros(1,7);
    
    nsamples = 6000;        % size of the monte carlo sample for muscle aproximation
    ntrials = 20;          % number of trials per epoch
    v = 5;                  % used for plotting data
    
    targAngles = -36:8:36;      % angles between eye and target locations
    angleSh = 0:0.05:pi/2;      % all possible theta1 values- shoulder
    angleEl = 0:0.05:pi/1.2;    % all possible theta2 values- elbow
    MA = [1 1 1 1 1];           % muscle activation output vector
    
    [ANGLE1, ANGLE2] = meshgrid(angleSh, angleEl);  % generate a grid of theta1 and theta2 values
    
    X = l1 * cos(ANGLE1) + l2 * cos(ANGLE1 + ANGLE2); % compute x coordinates
    Y = l1 * sin(ANGLE1) + l2 * sin(ANGLE1 + ANGLE2); % compute y coordinates
    
    count = 1;
    for interval=1:length(week_interval) % makes the entire program run with one click and stores the relevant data in one vector
        % set some variables pertaining to development based on current 'age'
        if week_interval(interval)==1
            MW = [max(angleSh) 1 max(angleEl) 1];     % muscle weights
            nepochs = 50;          % number of learning steps for the current week interval
            ratio = 3;              % signal to noise ratio for the current week interval
            epsilon_0 = 0.05;      % learning rate
        elseif week_interval(interval)==4
            MW = [max(angleSh) 1 max(angleEl) 1];     % muscle weights
            nepochs = 50;          % number of learning steps for the current week interval
            ratio = 6;              % signal to noise ratio for the current week interval
            epsilon_0 = 0.05;       % learning rate
        elseif week_interval(interval)==6
            MW = [max(angleSh) 1 max(angleEl) 1];     % muscle weights
            nepochs = 50;          % number of learning steps for the current week interval
            ratio = 8;              % signal to noise ratio for the current week interval
            epsilon_0 = 0.05;       % learning rate
        elseif week_interval(interval)==7
            MW = [max(angleSh) 1 max(angleEl) 1];     % muscle weights- antagonists will be activated from here on
            nepochs = 25;          % number of learning steps for the current week interval
            ratio = 8;              % signal to noise ratio for the current week interval
            epsilon_0 = 0.05;       % learning rate
        elseif week_interval(interval)==10
            MW = [max(angleSh) 1 max(angleEl) 1];     % muscle weights
            nepochs = 75;          % number of learning steps for the current week interval
            ratio = 10;              % signal to noise ratio for the current week interval
            epsilon_0 = 0.05;       % learning rate
        elseif week_interval(interval)==13
            MW = [max(angleSh) 1 max(angleEl) 1];     % muscle weights
            nepochs = 75;          % number of learning steps for the current week interval
            ratio = 10;              % signal to noise ratio for the current week interval
            epsilon_0 = 0.05;       % learning rate
        elseif week_interval(interval)==16
            MW = [max(angleSh) 1 max(angleEl) 1];     % muscle weights
            nepochs = 100;          % number of learning steps for the current week interval
            ratio = 10;              % signal to noise ratio for the current week interval
            epsilon_0 = 0.05;       % learning rate
        elseif week_interval(interval)==19
            MW = [max(angleSh) 1 max(angleEl) 1];     % muscle weights
            nepochs = 100;          % number of learning steps for the current week interval
            ratio = 10;              % signal to noise ratio for the current week interval
            epsilon_0 = 0.05;        % learning rate
        end
        
        % set network hyperparameters
        tau_0 = 10;
        tau_v = nepochs;
        epsilon_v = nepochs;
        
        samples = zeros(nsamples,4);            % matrix for storing muscle samples
        posSamples = zeros(nsamples,35);         % matrix for storing position samples resulting from muscle samples
        
        %% Create training and validation sets, and sample muscle data
        rng('shuffle');
        
        % sample Muscle activations and their resulting x and y positions
        for i = 1:nsamples;
            samples(i,:) = rand([4,1]);
            
            upperMov = (max(angleSh) * samples(i,1)) - (MW(2) * samples(i,2));       % movement factor of the upper arm
            foreMov = pi/1.2 - (max(angleEl) * samples(i,3) - MW(4) * samples(i,4)); % movement factor of the forearm
            
            upperMov = max(0,upperMov);     % bound the variables- prevents impossible joint angles
            upperMov = min(max(angleSh), upperMov);
            foreMov = max(0,foreMov);       % bound the variables- prevents impossible joint angles
            foreMov = min(max(angleEl), foreMov);
            
            posX = l1 * cos(upperMov) + l2 * cos(upperMov + foreMov); % compute hand x
            posY = l1 * sin(upperMov) + l2 * sin(upperMov + foreMov); % compute hand y
            
            distance = floor(sqrt((posX+5)^2 + posY^2));  % round distance down to integer
            angle = 90-radtodeg(atan(posY/abs(posX+5)));
            
            targetDistance = zeros(1,25);   % construct discrete distance vector
            targetAngle = zeros(1,10);                          % construct discrete angle vector
            
            % discretize input angle vector
            if angle > -36 && angle < 36
                index = 0;
                for j = 1:10
                    if (angle >= targAngles(j)-4) && (angle < targAngles(j)+4)
                        index = j;
                    end
                end
                if index > 0
                    targetAngle(index) = 1;
                end
            end
            
            % ensure target is within the visual field before providing depth information
            if sum(targetAngle)>0
                targetDistance(distance) = 1;   % discretize target distance
            end
            
            posSamples(i,1:25) = targetDistance;
            posSamples(i,26:35) = targetAngle;
            noise = randn(1,35)./ratio;
            posSamples(i,:)=posSamples(i,:)+noise;
        end
        
        % construct a training set
        trainingSet = zeros(2,nepochs*ntrials);
        for i = 1:1:length(trainingSet)
            rand1 = ceil(rand(1)*53);
            rand2 = ceil(rand(1)*32);
            
            targetx = X(rand1,rand2);
            targety = Y(rand1,rand2);
            trainingSet(1,i) = targetx;
            trainingSet(2,i) = targety;
        end
        
        % construct a validation set of only visible targets. Currently unused
        validationSet = zeros(2,nepochs/v+1);
        for i = 1:1:length(validationSet)
            rand1 = ceil(rand(1)*53);
            rand2 = ceil(rand(1)*32);
            
            targetx = X(rand1,rand2);
            targety = Y(rand1,rand2);
            distance = floor(sqrt((targetx+5)^2 + targety^2));
            angle = 90-radtodeg(atan(targety/abs(targetx+5)));
            if (angle < -36) && (angle > 36)
                i = max(i-1, 1);
            else
                validationSet(1,i) = targetx;
                validationSet(2,i) = targety;
            end
        end
        validationSet = validationSet(1:2,1:length(validationSet)-1);
        
        % construct test set
        testSet = zeros(2,144);
        pos = 1;
        while (testSet(1,144)==0)
            a = -15;
            b = 25;
            targetx = (b-a).*rand(1) + a;
            a = 10;
            b = 25;
            targety = (b-a).*rand(1) + a;
            distance = floor(sqrt((targetx)^2 + targety^2));
            if ((distance < 23) && (distance > 21))
                testSet(1,pos) = targetx;
                testSet(2,pos) = targety;
                pos = pos+1;
            end
        end
        
        clear rand1; clear rand2; clear targetx; clear targety; clear i;
        
        %% initialize network
        
        inputDistance = zeros(1,25);
        inputAngle = zeros(1,10);
        inputEyes = 0;
        
        imax_reach = length(inputDistance) + length(inputAngle);    % total input layer size
        jmax_reach = length(inputAngle)*4;                          % total hidden layer size
        kmax_reach = length(MA);                                    % total output layer size
        
        a = -0.5; b = 0.5;
        if week_interval(interval) == 1
            vij_reach = a + (b-a).*rand(imax_reach,jmax_reach); %initial weights input-hidden layer for newborn
            wjk_reach = a + (b-a).*rand(jmax_reach,kmax_reach); %initial weights hidden-output layer for newborn
        end
        
        % some vectors used for plotting results to provide some insight. Mostly
        % used during development of the code.
        dErrors = zeros(1,nepochs);
        mErrors = zeros(1,nepochs);
        vErrors = zeros(1,nepochs);
        positions = zeros(2,nepochs);
        if week_interval(interval)<6
            muscle = sum(gradient(muscle_errors(max(interval-1,1):interval)));
            visual = sum(gradient(fix_errors(max(interval-1,1):interval)));
            %epsilon_0 = (muscle+visual)/2;
        elseif week_interval(interval) >6
            muscle = sum(gradient(muscle_errors(max(interval-2,1):interval-1)));
            visual = sum(gradient(fix_errors(max(interval-2,1):interval-1)));
            %epsilon_0 = (muscle+visual)/2;
        end
        %epsilon_0 = max(abs(epsilon_0), 0.01);
        
        %% train network
        for j = 1:1:nepochs
            disp(interval)
            %epsilon_0 = ((sum(gradient(mErrors(max(j-1,1):j)))/2)+(sum(gradient(vErrors(max(j-1, 1):j)))/2))./2; %%MAKE BASIC
            %EPSILON THE CURRENT DERIVATIVE OF THE PERFORMANCES
            activations = zeros(ntrials,length(MA));
            nearest_activations = zeros(ntrials,length(MA));
            
            epsilons(count) = epsilon_0;
            count = count + 1;
            
            distances = zeros(1,ntrials);
            muscles = zeros(1,ntrials);
            visuals = zeros(1,ntrials);
            for trial = 1:1:ntrials
                input = [];
                hidden = [];
                output = [];
                tau = tau_0^((tau_v-j)/tau_v);
                epsilon = 10^(log10(epsilon_0)-j/epsilon_v);
                
                % load a randomized target position and discretize to distance
                targetx = trainingSet(1,j);     % target x
                targety = trainingSet(2,j);     % target y
                distance = floor(sqrt((targetx+5)^2 + targety^2));  % round distance down to integer
                angle = 90-radtodeg(atan(targety/abs(targetx+5)));
                targetAngle = zeros(1,10);                          % construct discrete angle vector
                
                %% perform a forward and backward pass in the motor network
                
                % discretize input angle vector
                if angle > -36 && angle < 36
                    index = 0;
                    for i = 1:10
                        if (angle >= targAngles(i)-4) && (angle < targAngles(i)+4)
                            index = i;
                        end
                    end
                    if index > 0
                        targetAngle(index) = 1;
                    end
                end
                
                targetDistance = zeros(1,25);   % construct discrete distance vector
                % ensure target is within the visual field before providing depth information
                if sum(targetAngle)>0
                    targetDistance(distance) = 1;   % discretize target distance
                end
                clear i; clear index;
                
                % apply Gaussian noise to inputs
                distnoise = randn(1,25)./ratio;         % Includes signal:noise ratio to crudely simulate growth of visual system
                anglenoise = randn(1,10)./ratio;        % Includes signal:noise ratio to crudely simulate growth of visual system
                tDistance = targetDistance+distnoise;
                tAngle = targetAngle+anglenoise;
                
                % construct input matrix from distance and direction vectors
                input = [tDistance tAngle];
                
                % calc output from input-hidden and hidden-output
                for i=1:jmax_reach
                    hidden(i) = sigmoid(vij_reach(:,i),input);
                end
                for k=1:kmax_reach
                    output(k) = weighted_sum(wjk_reach(:,k),hidden);
                end
                
                for i = 1:length(output)
                    output(i) = max(output(i),0.0001); % lower bound on muscle activation
                    output(i) = min(output(i),1); % upper bound on muscle activation
                end
                
                MA = output;
                
                % make the arm model move
                upperMov = (MW(1) * MA(1)) - (MW(2) * MA(2));        % movement factor of the upper arm
                foreMov = pi/1.2 - (MW(3) * MA(3) - MW(4) * MA(4));  % movement factor of the forearm
                
                upperMov = max(0,upperMov);     % bound the variables- prevents impossible joint angles
                upperMov = min(max(angleSh), upperMov);
                foreMov = max(0,foreMov);       % bound the variables- prevents impossible joint angles
                foreMov = min(max(angleEl), foreMov);
                
                x = l1 * cos(upperMov) + l2 * cos(upperMov + foreMov); % compute hand x
                y = l1 * sin(upperMov) + l2 * sin(upperMov + foreMov); % compute hand y
                
                % Store some information for debugging purposes- plotted afterwards
                positions(1,j) = x;
                positions(2,j) = y;
                
                xdist = abs(x-targetx);
                ydist = abs(y-targety);
                absdist = sqrt( xdist^2 + ydist^2);
                %errors(j)=mean(horzcat(errors(max(1,j-100):max(1,j-1)),absdist));
                
                % Determine best muscle activity combination from samples for current target
                pos = 1;
                mindist = 20;
                for i = 1:length(posSamples)
                    absdist = sum(abs(input - posSamples(i)));
                    if (absdist < mindist)
                        mindist = absdist;
                        nearestP = posSamples(i,:);
                        pos = i;
                    end
                end
                nearestMuscles = samples(pos,:);
                
                EA = MA(5);
                inputEyes = (EA-0.5)*40;   % angle of the eyes, bounded between -20 and 20 degrees from the center.
                angle = 90-radtodeg(atan(targety/abs(targetx+5)))-inputEyes;  % fit input angle to current visual field
                targetAngle = zeros(1,10);                                    % construct discrete angle vector
                
                bestAngle = max((angle/20), -1);        % lower bound on optimal eye muscle activation
                bestAngle = min(bestAngle, 1);          % upper bound on optimal eye muscle activation
                centeredView = bestAngle;
                nearestMuscles = [nearestMuscles centeredView];
                
                activations(trial,:) = MA;
                nearest_activations(trial,:) = nearestMuscles;
                
                % Calculate total error values to provide insight into backprop later
                % on for Code development purposes.
                div = 1/length(MA);
                distances(trial) = absdist;
                muscles(trial) = abs(div*((MA(1)-nearestMuscles(1))) + div*((MA(2)-nearestMuscles(2))) + div*((MA(3)-nearestMuscles(3))) + div*((MA(4)-nearestMuscles(4))));
                visuals(trial) = abs(div*((MA(5)-nearestMuscles(5))));
                
            end
            %change weights by mini-batch backprop; if ntrials = 1, the
            %mini-batches are size 1, and thus, this is stochastic backprop.
            
            MA = mean(activations,1);
            nearestMuscles = mean(nearest_activations,1);
            for k_ind=1:length(output)
                outo1=MA(k_ind);
                for k=1:jmax_reach
                    outh1= hidden(k);
                    for i=1:length(input)
                        var = 0;
                        i1 = input(i);
                        for o=1:length(output)
                            d0 = (nearestMuscles(o) - MA(o))*MA(o)*(1-MA(o));
                            who = wjk_reach(k,o);
                            var = var+(-d0*who);
                        end
                        var = var*outh1*(1-outh1)*i1;
                        vij_reach(i,k) = vij_reach(i,k) - epsilon*var;
                    end
                    change = (-((nearestMuscles(k_ind) - outo1))*outo1*(1-outo1))*outh1;
                    wjk_reach(k,k_ind) = wjk_reach(k,k_ind) - epsilon * change;
                end
                
            end
            
            dErrors(j) = mean(distances);
            mErrors(j) = mean(muscles);
            vErrors(j) = mean(visuals);
            
        end
        clear j;
        if week_interval(interval) < 6
            dist_errors(interval) = mean(dErrors);
            muscle_errors(interval) = mean(mErrors);
            fix_errors(interval) = mean(vErrors);
        elseif week_interval(interval)> 6
            dist_errors(interval-1) = mean(dErrors);
            muscle_errors(interval-1) = mean(mErrors);
            fix_errors(interval-1) = mean(vErrors);
        end
        %% Run on test set
        reaches = 0;
        fixreaches = 0;
        unfixreaches = 0;
        for j = 1:1:length(testSet)
            input1 = [];
            hidden1 = [];
            hiddenview1 = [];
            output1 = [];
            
            % load a target position and discretize to distance
            targetx = testSet(1,j);     % target x
            targety = testSet(2,j);     % target y
            
            distance = floor(sqrt((targetx)^2 + targety^2));    % round distance down to integer
            angle = 90-radtodeg(atan(targety/abs(targetx+5)));  % fit input angle to visual field
            targetAngle = zeros(1,10);                          % construct discrete angle vector
            
            %% perform a forward pass in the motor network
            angle = 90-radtodeg(atan(targety/abs(targetx+5)))-inputEyes;  % fit input angle to current visual field
            targetAngle = zeros(1,10);
            
            % discretize input angle vector
            if angle > -36 && angle < 36
                index = 0;
                for i = 1:10
                    if (angle > targAngles(i)-4) && (angle < targAngles(i)+4)
                        index = i;
                    end
                end
                if index > 0
                    targetAngle(index) = 1;
                end
            end
            targetDistance = zeros(1,25);   % construct discrete distance vector
            % ensure target is within the visual field before providing depth information
            if sum(targetAngle)>0
                targetDistance(distance) = 1;   % discretize target distance
            end
            clear i; clear index;
            
            % apply Gaussian noise to inputs
            distnoise = randn(1,25)./ratio;         % Includes signal:noise ratio to crudely simulate growth of visual system
            anglenoise = randn(1,10)./ratio;        % Includes signal:noise ratio to crudely simulate growth of visual system
            tDistance = targetDistance+distnoise;
            tAngle = targetAngle+anglenoise;
            
            % construct input matrix from distance and direction vectors
            input1 = [tDistance tAngle];
            
            % calc output from input-hidden and hidden-output
            for i=1:jmax_reach
                hidden1(i) = sigmoid(vij_reach(:,i),input1);
            end
            for k=1:kmax_reach
                output1(k) = weighted_sum(wjk_reach(:,k),hidden1);
            end
            
            for i = 1:length(output1)
                output1(i)  = abs(output1(i));
            end
            MA = output1;
            EA = MA(5);
            inputEyes = (EA-0.5)*40;   % angle of the eyes, bounded between -20 and 20 degrees from the center.
            angle = 90-radtodeg(atan(targety/abs(targetx+5)))-inputEyes;  % fit input angle to current visual field
            targetAngle = zeros(1,10);                                    % construct discrete angle vector
            
            % make the arm model move
            upperMov = MW(1) * MA(1) - MW(2) * MA(2);           % movement bias of the upper arm
            foreMov = pi/1.2 - (MW(3) * MA(3) - MW(4) * MA(4)); % movement bias of the forearm
            
            upperMov = max(0,upperMov);     % bound the variables- prevents impossible joint angles
            upperMov = min(max(angleSh), upperMov);
            foreMov = max(0,foreMov);       % bound the variables- prevents impossible joint angles
            foreMov = min(max(angleEl), foreMov);
            
            x = l1 * cos(upperMov) + l2 * cos(upperMov + foreMov); % compute hand x
            y = l1 * sin(upperMov) + l2 * sin(upperMov + foreMov); % compute hand y
            
            positions2(1,j) = x;
            positions2(2,j) = y;
            %   % Legacy code used for copying von Hofsten's methodology. Currently
            %   unused, may prove useful
            %     c = sqrt(x^2 + y^2);    % compute reach length
            
            % Count number of reaches exceeding a certain forward distance. This is
            % the main measurement used to compare to von Hosten's study, and is
            % what we measure for each week interval.
            if y >= reach_thresh
                if angle < 36 && angle > -36
                    fixreaches = fixreaches+1;
                else
                    unfixreaches = unfixreaches +1;
                end
                reaches = reaches +1;
            end
        end
        
        if not(week_interval(interval)==6)
            if week_interval(interval>6)
                fixated_reaches(1,interval-1) = fixreaches;
                unfixated_reaches(1,interval-1) = unfixreaches;
                extended_reaches(1,interval-1) = reaches;
            else
                fixated_reaches(1,interval) = fixreaches;
                unfixated_reaches(1,interval) = unfixreaches;
                extended_reaches(1,interval) = reaches;
            end
            
        end
    end
    
    %% plot possible positions and current position. Used for code development and debugging, but may be interesting to see.
    
    X = l1 * cos(ANGLE1) + l2 * cos(ANGLE1 + ANGLE2); % compute x coordinates
    Y = l1 * sin(ANGLE1) + l2 * sin(ANGLE1 + ANGLE2); % compute y coordinates
    
    
    % % plots the training targets (red), test targets (blue), last presented
    % % target (circle), and last reach position (square)
    % figure(1);
    % plot(X(:), Y(:), 'r.', x,y,'ko', targetx,targety,'ks',testSet(1,:),testSet(2,:),'b.');
    % axis equal;
    % xlabel('X','fontsize',10)
    % ylabel('Y','fontsize',10)
    
    dist_errors = cumsum(dist_errors);
    muscle_errors = cumsum(muscle_errors);
    fix_errors = cumsum(fix_errors);
    
    % plots the distance error over time
    figure(1);
    title('reach error');
    plot(dist_errors)
    xlabel('epochs','fontsize',10)
    ylabel('distance','fontsize',10)
    
    % plots the mean muscle error over time.
    figure(2);
    title('mean reaching error');
    plot(muscle_errors);
    xlabel('epochs','fontsize',10)
    ylabel('muscle error','fontsize',10)
    
    % plots the mean distance error over time.
    figure(3);
    title('mean fixation error');
    plot(fix_errors);
    xlabel('epochs','fontsize',10)
    ylabel('fixation error','fontsize',10)
    
    % plots the progression of one selected weight over time. Not very useful,
    % but interesting to see.
    figure(4);
    title('motor error derivative');
    plot((diff(muscle_errors)));
    xlabel('epochs','fontsize',10)
    ylabel('muscle error differential','fontsize',10)
    
    % plots the reach positions. Used for debugging the motor system
    plottable = diff(fix_errors);
    indis = length(plottable)-5;
    figure(5);
    title('fixation error derivative');
    %plot(plottable(indis:length(plottable))+1);
    plot(plottable)
    xlabel('epochs','fontsize',10)
    ylabel('fixation error differential','fontsize',10)
    
    figure(6);
    title('learning rate over time');
    plot(epsilons)
    xlabel('epochs','fontsize',10)
    ylabel('epsilon','fontsize',10)
    
    
    % plots the extended reaches for all week intervals
    figure(7);
    title('number of fixated and unfixated reaches');
    plot(1:length(fixated_reaches),fixated_reaches(1,:),'r',1:length(unfixated_reaches), unfixated_reaches, 'b')
    set(gca, 'XTickLabel',{'1','4','7','10','13','16','19'})
    
    % plots the extended reaches for all week intervals
    figure(8);
    title('number of extended reaches');
    plot(extended_reaches(1,:))
    set(gca, 'XTickLabel',{'1','4','7','10','13','16','19'})
    
    results_exReaches(experiment,:) = extended_reaches;
    results_fixReaches(experiment,:) = fixated_reaches;
    results_unfixReaches(experiment,:) = unfixated_reaches;
    results_derivFix(experiment,:) = diff(fix_errors);
    results_derivMusc(experiment,:) = diff(muscle_errors);
end
%%
csvwrite('Data/flat_extensions2.csv',results_exReaches);
%type extensions.dat;
csvwrite('Data/flat_fixated_extensions2.csv',results_fixReaches);
%type fixated_extensions.dat;
csvwrite('Data/flat_unfixated_extensions2.csv',results_unfixReaches);
%type unfixated_extensions.dat;
csvwrite('Data/flat_fixation_derivative2.csv',results_derivFix);
%type fixation_derivative.dat;
csvwrite('Data/flat_muscle_derivative2.csv',results_derivMusc)
%type muscle_derivative.dat;