
function [activation] = input_gauss_activation(data,centers,radii)


for i=1:length(data)
    input = data(i);

    for c=1:length(centers) 
        sigma = radii(c)^2;
        diff = input - centers(c);
    
        activation(i,c) = exp(-diff*diff/sigma);
    end
end  
