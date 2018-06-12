
function [yj] = sigmoid(vij, xi)

sum_activation = 0;

for i=1:length(xi)
    sum_activation = sum_activation + xi(i)*vij(i);
end

yj = (pi/2)/(1 + exp(-sum_activation));