clear ; close all; clc

n =3;
m = 100; 
data = load('ex2data1.txt');

X = data(:, [1, 2]);
X = [ones(m, 1) X];
y = data(:, 3);

theta = zeros(n , 1);
grad = zeros(size(theta));

sum = 0;
for i = 1:m
    sum = sum + (-y(i)*log(1./(1+exp(theta*X(i, :)))) - (1-y(i))*log(1-(1+exp(theta*X(i, :)))));
end 

for j =1:n
    temp = 0;
    for i = 1:m
        temp  = temp + (1./(1+exp(theta*X(i, :)))-y(i))*X(i, j); 
    end 
    grad(j) = temp/m;
end

