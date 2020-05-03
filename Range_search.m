clear;
close all;
clc;

dt = 1;
time = 0.0;
endtime = 50.0;
nsteps = ceil((endtime - time) / dt);

result.x = [];
result.y = [];
result.time = [];

x = 10;
w = 0.5^2;
v = 0.05^2;
a = -25;
b = 25;
N = 400;
% データセット
% 一般に、区間 (a,b) の N 個の乱数は、式 r = a + (b-a).*rand(N,1) を使って生成できます。
X_t = a + (b - a).*rand(N, 1);
result.X_t_1 = [];
result.Y_t   = [];
for i = 1:N
    X_t_1 = 0.2 * X_t(i, 1) + ((25 * X_t(i, 1)) / (1 + X_t(i, 1)^2)) + 8 * cos(1.2 * X_t(i, 1)) +  sqrt(w) * randn(1, 1);
    Y_t   = sin(X_t(i, 1) / 10) + sqrt(v) * randn(1, 1);
    result.X_t_1 = [result.X_t_1; X_t_1];
    result.Y_t   = [result.Y_t; Y_t];
end

result.K    = [];
result.temp = [];
alpha_f = 1.937;
beta_f  = 0.400;
lamda_f = 4.126;
for i = 1:N
    for j = 1:N
        k = alpha_f^2 * exp((-1 / 2)) * beta_f * (X_t(j, 1) -  X_t(i));
        result.temp = [result.temp; k];
    end
    result.K = vertcat(result.K, result.temp');
    result.temp = [];
end
% 平均
m = result.K(1,:) / (result.K + lamda_f * eye(N, N)) * result.X_t_1;
% 分散
sigma = alpha_f^2 - result.K(1,:) / (result.K + lamda_f * eye(N, N)) * (result.K(1,:))';
tic;% start　for文回る時間の計測
for i = 1:nsteps
    time = time + dt;
    x = 0.2 * x + ((25 * x) / (1 + x^2)) + 8 * cos(1.2 * x) +  sqrt(w) * randn(1, 1);
    y = sin(x / 10) + sqrt(v) * randn(1, 1);
    result.x = [result.x; x];
    result.y = [result.y; y];
    result.time = [result.time; time];
end
toc;% end
Drow(result);

function [] = Drow(result)
    figure(1);
    plot(result.time, result.x, 'k');
    figure(2);
    plot(result.time, result.y, 'k');
end