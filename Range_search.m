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
theta = -0.05;
%% ハイパーパラメータの値
% alpha_f = 1.937;
% beta_f  = 0.400;
% lamda_f = 4.126;
% alpha_h = 0.999;
% beta_h  = 0.002;
% lamda_h = 0.198;

alpha_f = 2.8989 * 10^7;
beta_f  = 4.2002;
lamda_f = 1.8627 * 10^4;
alpha_h = 138.3217;
beta_h  = 0.2;
lamda_h = 9.2576 * 10^3;
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
%% 初期値設定
result.x     = [result.x; x];
result.time  = [result.time; time];
result.xEst  = [result.xEst; xEst];
result.PEst  = [result.PEst; PEst];
%% 関数の平均値を格納
result.m     = [];
%% 関数の分散を格納
result.sigma = [];
result.Upper = [];
result.Lower = [];
%% データセット生成
[X_t, X_t_1, Y_t, K, L] = makeDataset(a, b, w, v, alpha_f, alpha_h, beta_f, beta_h, N);
m = K(1,:) / (K + lamda_f * eye(N, N)) * X_t_1;
% 分散
sigma = alpha_f^2 - K(1,:) / (K + lamda_f * eye(N, N)) * (K(1,:))';
Upper = m + 3 * sqrt(sigma);
Lower = m - 3 * sqrt(sigma);
%% データセットから推定した関数の平均と分散を初期値として配列に格納
result.m     = [result.m; m];
result.sigma = [result.sigma; sigma];
result.Upper = [result.Upper; Upper];
result.Lower = [result.Lower; Lower];
%% main loop
tic;% start　
for i = 1:nsteps
    time = time + dt;
    % ----ガウス過程回帰による関数fの推定-----
    % k*(x_t)の計算 式(4)
    k_star = Calc_k_star(N, alpha_f, beta_f, X_t, x);
    % 平均 式(8)
    m = (k_star)' / (K + lamda_f * eye(N, N)) * X_t_1;
    % 分散 式(8)
    sigma = alpha_f^2 - (k_star)' / (K + lamda_f * eye(N, N)) * k_star;
    Upper = m + 3 * sqrt(sigma);
    Lower = m - 3 * sqrt(sigma);
    %% 関数値の計算
    x = 0.2 * x + ((25 * x) / (1 + x^2)) + 8 * cos(1.2 * x) +  sqrt(w) * randn(1, 1);
    y = sin(x / 10) + sqrt(v) * randn(1, 1);
    %---------------------------------------

    %% ------ リスク鋭敏型フィルタ -------
    %% 予測ステップ
    % m_fにxEstを入れた値を計算するためにk*(x_t)の計算
    k_star_xPred = Calc_k_star(N, alpha_f, beta_f, X_t, xEst);
    % m_fにxEstを入れた値をm_xEstとしている
    m_xEst = (k_star_xPred)' / (K + lamda_f * eye(N, N)) * X_t_1;
    xPred  = m_xEst; % 式(16)
    % ヤコビアンm_Fのためのk*(x_t)の偏微分の計算
    k_star_mf = Calc_k_star_mf(N, alpha_f, beta_f, X_t, xPred);
    % ヤコビアンm_F
    m_F   = (k_star_mf)' / (K + lamda_f * eye(N, N)) * X_t_1; % 式(10)
    PPred = m_F * PEst * m_F' + w;
    %% フィルタリングステップ
    % ヤコビアンm_Hのためのk*(x_t)の偏微分の計算
    k_star_mh = Calc_k_star_mf(N, alpha_h, beta_h, X_t, xPred);
    % ヤコビアンm_H
    m_H = (k_star_mh)' / (K + lamda_h * eye(N, N)) * Y_t; % 式(12)
    %カルマンゲイン
    G = PPred * m_H' / (m_H * PPred * m_H' + v); % 式(18)
    % 状態推定値
    % m_hにxPredを入れた値を求めるためにk*(x_t)の計算
    k_star_xEst = Calc_k_star(N, alpha_h, beta_h, X_t, xPred);
    % 平均
    m_h  = (k_star_xEst)' / (K + lamda_h * eye(N, N)) * Y_t;
    xEst = xPred + G * (y - m_h); % 式(17)
    % 事後誤差共分散行列
    % ヤコビアンmfにxPredを入れた値を求めるためにk*(x_t)の偏微分の計算
    k_star_xEst_last = Calc_k_star_mf(N, alpha_f, beta_f, X_t, xEst);
    % ヤコビアンmfにxPredを入れた値を計算
    m_F  = (k_star_xEst_last)' / (K + lamda_f * eye(N, N)) * X_t_1; % 式(10)
    PEst = m_F^2 / (PPred + m_H^2 / v + theta) + w; % 式(19)
    % thetaの条件式
    if PPred + m_H^2 / v + theta <= 0
        break;
    end
    %---------------------------------------

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

function [X_t, X_t_1, Y_t, K, L] = makeDataset(a, b, w, v, alpha_f, alpha_h, beta_f, beta_h, N)
    % データセットと状態方程式を基にX_t+1とY_tのデータセットを生成
    % 一般に、区間 (a,b) の N 個の乱数は、式 r = a + (b-a).*rand(N,1) を使って生成できます。
    X_t   = a + (b - a).*rand(N, 1);
    X_t_1 = zeros(N, 1); 
    Y_t   = zeros(N, 1);
    for i = 1:N
        X_t_1(i, 1) = 0.2 * X_t(i, 1) + ((25 * X_t(i, 1)) / (1 + X_t(i, 1)^2)) + 8 * cos(1.2 * X_t(i, 1)) +  sqrt(w) * randn(1, 1);
        Y_t(i, 1)   = sin(X_t(i, 1) / 10) + sqrt(v) * randn(1, 1);
    end
    % データセットを基に関数fの平均と分散を計算
    K = zeros(N, N);
    for i = 1:N
        for j = 1:N
            K(i, j) = alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  X_t(i))^2);
        end
    end
    % データセットを基に関数hの平均と分散を計算
    L = zeros(N, N);
    for i = 1:N
        for j = 1:N
            L(i, j) = alpha_h^2 * exp((-1 / 2) * beta_h * (X_t(j, 1) -  X_t(i))^2);
        end
    end
end

function k_star = Calc_k_star(N, alpha_f, beta_f, X_t, x)
    k_star = zeros(N, 1);
    for j = 1:N
        k_star(j, 1) = alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  x)^2);
    end
end

function k_star = Calc_k_star_mf(N, alpha_f, beta_f, X_t, x)
    k_star = zeros(N, 1);
    for j = 1:N
        k_star(j, 1) = -beta_f * (X_t(j, 1) -  x) * alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  x)^2);
    end
end
    
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