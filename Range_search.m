clear;
close all;
clc;
%% シミュレーション時間設定
dt      = 1;
time    = 0.0;
endtime = 50.0;
nsteps  = ceil((endtime - time) / dt);
%% 
% フィルタ結果格納
result.x    = [];
result.y    = [];
result.time = [];
result.xEst = [];
result.PEst = [];
% 初期ガウス過程回帰の行列Kと一時保存用配列temp
result.K    = [];
result.temp = [];
x    = 10;
xEst = 10;
PEst = 1;
w = 0.5^2;
v = 0.05^2;
a = -20;
b = 20;
N = 500;
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

alpha_f = 1.937;
beta_f  = 0.400;
lamda_f = 4.126;
alpha_h = 0.999;
beta_h  = 0.002;
lamda_h = 0.198;
for i = 1:N
    for j = 1:N
        k = alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  X_t(i))^2);
        result.temp = [result.temp; k];
    end
    result.K = vertcat(result.K, result.temp');
    result.temp = [];
end
% 平均
m = result.K(1,:) / (result.K + lamda_f * eye(N, N)) * result.X_t_1;
% 分散
sigma = alpha_f^2 - result.K(1,:) / (result.K + lamda_f * eye(N, N)) * (result.K(1,:))';
Upper = m + 3 * sqrt(sigma);
Lower = m - 3 * sqrt(sigma);

result.Temp_x  = [];
result.Temp_xEst  = [];
result.Temp_xEst_y  = [];
result.Temp_xEst_last  = [];
result.Temp_xPred_x  = [];
result.Temp_xPred_y = [];
result.m     = [];
result.m_xEst = [];
result.m     = [result.m; m];
result.m_xEst = [result.m_xEst; m];
result.sigma = [];
result.sigma = [result.sigma; sigma];
result.Upper = [];
result.Upper = [result.Upper; Upper];
result.Lower = [];
result.Lower = [result.Lower; Lower];
result.time  = [result.time; time];
result.x     = [result.x; x];
result.xEst  = [result.xEst; xEst];
result.PEst  = [result.PEst; PEst];
tic;% start　for文回る時間の計測
for i = 1:nsteps
    time = time + dt;
    % k*(x_t)の計算
    for j = 1:N
        k = alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  x)^2);
        result.Temp_x = [result.Temp_x; k];
    end
    % 平均
    m =  (result.Temp_x)' / (result.K + lamda_f * eye(N, N)) * result.X_t_1;
    % 分散
    sigma = alpha_f^2 - (result.Temp_x)' / (result.K + lamda_f * eye(N, N)) * result.Temp_x;
    Upper = m + 3 * sqrt(sigma);
    Lower = m - 3 * sqrt(sigma);
    if time >= 25 && time <= 30 
        % 関数値の計算
        x = 0.2 * x + ((25 * x) / (1 + x^2)) + 8 * cos(1.2 * x) +  10 * sqrt(w) * randn(1, 1);
        y = sin(x / 10) + 10 * sqrt(v) * randn(1, 1);
    else
        % 関数値の計算
        x = 0.2 * x + ((25 * x) / (1 + x^2)) + 8 * cos(1.2 * x) +  sqrt(w) * randn(1, 1);
        y = sin(x / 10) + sqrt(v) * randn(1, 1);
    end
    % ------リスク鋭敏型フィルタ --------
    % 予測ステップ
    % k*(x_t)の計算
    for j = 1:N
        k = alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  xEst)^2);
        result.Temp_xEst = [result.Temp_xEst; k];
    end
    % 平均
    m_xEst =  (result.Temp_xEst)' / (result.K + lamda_f * eye(N, N)) * result.X_t_1;
    xPred = m_xEst; % 式(16)
    % k*(x_t)の偏微分の計算
    for j = 1:N
        k = -beta_f * (X_t(j, 1) -  xPred) * alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  xPred)^2);
        result.Temp_xPred_x = [result.Temp_xPred_x; k];
    end
    % ヤコビアンF(m_F)
    F     = (result.Temp_xPred_x)' / (result.K + lamda_f * eye(N, N)) * result.X_t_1; % 式(10)
    PPred = F * PEst * F' + w;

    % フィルタリングステップ
    % k*(x_t)の偏微分の計算
    for j = 1:N
        k = -beta_h*(X_t(j, 1) -  xPred)*alpha_h^2 * exp((-1 / 2) * beta_h * (X_t(j, 1) -  xPred)^2);
        result.Temp_xPred_y = [result.Temp_xPred_y; k];
    end
    % ヤコビアンH(m_H)
    H     = (result.Temp_xPred_y)' / (result.K + lamda_h * eye(N, N)) * result.Y_t; % 式(12)
    %カルマンゲイン
    K = PPred*H' / (H*PPred*H' + v); % 式(18)
    %状態推定値
    % k*(x_t)の計算
    for j = 1:N
        k = alpha_h^2 * exp((-1 / 2) * beta_h * (X_t(j, 1) -  xPred)^2);
        result.Temp_xEst_y = [result.Temp_xEst_y; k];
    end
    % 平均
    m_h =  (result.Temp_xEst_y)' / (result.K + lamda_h * eye(N, N)) * result.Y_t;
    xEst = xPred + K * (y - m_h); % 式(17)
    %事後誤差共分散行列
    % k*(x_t)の偏微分の計算
    for j = 1:N
        k = -beta_f * (X_t(j, 1) -  xEst) * alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  xEst)^2);
        result.Temp_xEst_last = [result.Temp_xEst_last; k];
    end
    
    % ヤコビアンF
    F     = (result.Temp_xEst_last)' / (result.K + lamda_f * eye(N, N)) * result.X_t_1; % 式(10)
    theta = 0.05;
    PEst  = F^2 / (PPred + H^2 / v + theta) + w; % 式(19)



    result.x = [result.x; x];
    result.y = [result.y; y];
    result.time = [result.time; time];
    result.xEst  = [result.xEst; xEst];
    result.Temp = [];
    result.Temp_x  = [];
    result.Temp_xEst  = [];
    result.Temp_xEst_y  = [];
    result.Temp_xEst_last  = [];
    result.Temp_xPred_x  = [];
    result.Temp_xPred_y = [];
    result.m = [result.m; m];
    result.sigma = [result.sigma; sigma];
    result.Upper = [result.Upper; Upper];
    result.Lower = [result.Lower; Lower];
end
toc;% end
Drow(result);

function [] = Drow(result)
    figure(1);
    plot(result.time, result.x, 'k'); hold on;
    plot(result.time, result.Upper, 'r');hold on;
    plot(result.time, result.m, 'g'); hold on;
    plot(result.time, result.Lower, 'b');hold on;
    figure(2);
    plot(result.time, result.x, 'k'); hold on;
    plot(result.time, result.xEst, 'r');hold on;
end