
function [ok] = weighted_sum(wjk, hidden)

sum_activation = 0;

for j=1:length(hidden)
    sum_activation = sum_activation + wjk(j)*hidden(j);
end

ok = sum_activation;