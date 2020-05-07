clear;
close all;
clc;
%% シミュレーション時間設定
dt      = 1;
time    = 0.0;
endtime = 50.0;
nsteps  = ceil((endtime - time) / dt);
%% 状態の初期値
x     = 10;
xEst  = 10;
PEst  = 1;
w     = 0.5^2;
v     = 0.05^2;
theta = 0.05;
%% ハイパーパラメータの値
alpha_f = 1.937;
beta_f  = 0.400;
lamda_f = 4.126;
alpha_h = 0.999;
beta_h  = 0.002;
lamda_h = 0.198;
%% データセット取得範囲及び個数
a = -20;
b = 20;
N = 500;
%% フィルタ結果格納
result.x    = [];
result.y    = [];
result.time = [];
result.xEst = [];
result.PEst = [];
% 初期値設定
result.x     = [result.x; x];
result.time  = [result.time; time];
result.xEst  = [result.xEst; xEst];
result.PEst  = [result.PEst; PEst];
%% 関数の平均値を格納
result.m      = [];
%% 関数の分散を格納
result.sigma = [];
result.Upper = [];
result.Lower = [];
%% データセットと状態方程式を基にX_t+1とY_tのデータセットを生成
% 一般に、区間 (a,b) の N 個の乱数は、式 r = a + (b-a).*rand(N,1) を使って生成できます。
X_t   = a + (b - a).*rand(N, 1);
X_t_1 = zeros(N, 1); 
Y_t   = zeros(N, 1);
for i = 1:N
    X_t_1(i, 1) = 0.2 * X_t(i, 1) + ((25 * X_t(i, 1)) / (1 + X_t(i, 1)^2)) + 8 * cos(1.2 * X_t(i, 1)) +  sqrt(w) * randn(1, 1);
    Y_t(i, 1)   = sin(X_t(i, 1) / 10) + sqrt(v) * randn(1, 1);
end
%% データセットを基に関数fの平均と分散を計算
K = zeros(N, N);
for i = 1:N
    for j = 1:N
        K(i, j) = alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  X_t(i))^2);
    end
end
%% データセットを基に関数hの平均と分散を計算
L = zeros(N, N);
for i = 1:N
    for j = 1:N
        L(i, j) = alpha_h^2 * exp((-1 / 2) * beta_h * (X_t(j, 1) -  X_t(i))^2);
    end
end
m = K(1,:) / (K + lamda_f * eye(N, N)) * X_t_1;
% 分散
sigma = alpha_f^2 - K(1,:) / (K + lamda_f * eye(N, N)) * (K(1,:))';
Upper = m + 3 * sqrt(sigma);
Lower = m - 3 * sqrt(sigma);
%% データセットから推定した関数の平均と分散を初期値として配列に格納
result.m      = [result.m; m];
result.sigma  = [result.sigma; sigma];
result.Upper  = [result.Upper; Upper];
result.Lower  = [result.Lower; Lower];
%% main loop
tic;% start　
k_star = zeros(N, 1);
for i = 1:nsteps
    time = time + dt;
    %% ガウス過程回帰による関数fの推定
    % k*(x_t)の計算 式(4)
    for j = 1:N
        k_star(j, 1) = alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  x)^2);
    end
    % 平均 式(8)
    m =  (k_star)' / (K + lamda_f * eye(N, N)) * X_t_1;
    % 分散 式(8)
    sigma = alpha_f^2 - (k_star)' / (K + lamda_f * eye(N, N)) * k_star;
    Upper = m + 3 * sqrt(sigma);
    Lower = m - 3 * sqrt(sigma);
    %% 関数値の計算
    x = 0.2 * x + ((25 * x) / (1 + x^2)) + 8 * cos(1.2 * x) +  sqrt(w) * randn(1, 1);
    y = sin(x / 10) + sqrt(v) * randn(1, 1);
    %% 配列に結果を格納
    result.x     = [result.x; x];
    result.y     = [result.y; y];
    result.time  = [result.time; time];
    result.xEst  = [result.xEst; xEst];
    result.m     = [result.m; m];
    result.sigma = [result.sigma; sigma];
    result.Upper = [result.Upper; Upper];
    result.Lower = [result.Lower; Lower];
end
toc;

Drow(result);

function [] = Drow(result)
    figure(1);
    plot(result.time, result.x, 'k'); hold on;
    plot(result.time, result.Upper, 'r');hold on;
    plot(result.time, result.m, 'g'); hold on;
    plot(result.time, result.Lower, 'b');hold on;
end