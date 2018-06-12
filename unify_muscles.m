%% function for determining the net force in favor of forward movement. 
%  TODO: - Allow for non-linear trade-off between muscles
%           * Maybe several options, just as parameter?
function act = unify_muscles(ag,ant,coac)
r = (rand(1)-0.5)/20; % generate a random noise variable between -0.5 and 0.5, and scale it down
rc = r/sqrt(coac);    % scale the noise by the coactivation coefficient: more coac is less noise
a = ag-ant*coac;      % determine base net muscle activation
act = a+a*rc;         % muscle output gets added noise proportional to the movement size.