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
theta = 0.01;
%% ハイパーパラメータの値
% alpha_f = 1.937;
% beta_f  = 0.400;
% lamda_f = 4.126;
% alpha_h = 0.999;
% beta_h  = 0.002;
% lamda_h = 0.198;

% alpha_f = 4.7023 * 10^7;
% beta_f  = 4.2;
% lamda_f = 9.63 * 10^3;
% alpha_h = 895;
% beta_h  = 0.2;
% lamda_h = 8.02 * 10^3;

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
% 初期値設定
result.x     = [result.x; x];
result.time  = [result.time; time];
result.xEst  = [result.xEst; xEst];
result.PEst  = [result.PEst; PEst];
%% 初期ガウス過程回帰の行列Kと一時保存用配列temp
result.K    = [];
result.temp = [];
%% データセット格納配列
result.X_t_1 = [];
result.Y_t   = [];
%% m_f,m_h,m_F,m_Hの計算過程で使用する一時保存用配列
result.Temp_x         = [];
result.Temp_xEst      = [];
result.Temp_xEst_y    = [];
result.Temp_xEst_last = [];
result.Temp_xPred_x   = [];
result.Temp_xPred_y   = [];
%% 関数の平均値を格納
result.m      = [];
result.m_xEst = [];
%% 関数の分散を格納
result.sigma = [];
result.Upper = [];
result.Lower = [];
%% データセットと状態方程式を基にX_t+1とY_tのデータセットを生成
% 一般に、区間 (a,b) の N 個の乱数は、式 r = a + (b-a).*rand(N,1) を使って生成できます。
X_t = a + (b - a).*rand(N, 1);
for i = 1:N
    X_t_1 = 0.2 * X_t(i, 1) + ((25 * X_t(i, 1)) / (1 + X_t(i, 1)^2)) + 8 * cos(1.2 * X_t(i, 1)) +  sqrt(w) * randn(1, 1);
    Y_t   = sin(X_t(i, 1) / 10) + sqrt(v) * randn(1, 1);
    result.X_t_1 = [result.X_t_1; X_t_1];
    result.Y_t   = [result.Y_t; Y_t];
end
%% データセットを基に関数fの平均と分散を計算
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
Upper = m + sqrt(sigma);
Lower = m - sqrt(sigma);
%% データセットから推定した関数の平均と分散を初期値として配列に格納
result.m      = [result.m; m];
result.m_xEst = [result.m_xEst; m];
result.sigma  = [result.sigma; sigma];
result.Upper  = [result.Upper; Upper];
result.Lower  = [result.Lower; Lower];
%% main loop
tic;% start　
for i = 1:nsteps
    time = time + dt;
    %% ガウス過程回帰による関数fの推定
    % k*(x_t)の計算 式(4)
    for j = 1:N
        k = alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  x)^2);
        result.Temp_x = [result.Temp_x; k];
    end
    % 平均 式(8)
    m =  (result.Temp_x)' / (result.K + lamda_f * eye(N, N)) * result.X_t_1;
    % 分散 式(8)
    sigma = alpha_f^2 - (result.Temp_x)' / (result.K + lamda_f * eye(N, N)) * result.Temp_x;
    Upper = m + 3 * sqrt(sigma);
    Lower = m - 3 * sqrt(sigma);
    %% 関数値の計算
    x = 0.2 * x + ((25 * x) / (1 + x^2)) + 8 * cos(1.2 * x) +  sqrt(w) * randn(1, 1);
    y = sin(x / 10) + sqrt(v) * randn(1, 1);
    %% ------リスク鋭敏型フィルタ --------
    %% 予測ステップ
    % m_fにxEstを入れた値を計算するためにk*(x_t)の計算
    for j = 1:N
        k = alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  xEst)^2);
        result.Temp_xEst = [result.Temp_xEst; k];
    end
    % m_fにxEstを入れた値をm_xEstとしている
    m_xEst =  (result.Temp_xEst)' / (result.K + lamda_f * eye(N, N)) * result.X_t_1;
    xPred = m_xEst; % 式(16)
    % ヤコビアンm_Fのためのk*(x_t)の偏微分の計算
    for j = 1:N
        k = -beta_f * (X_t(j, 1) -  xPred) * alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  xPred)^2);
        result.Temp_xPred_x = [result.Temp_xPred_x; k];
    end
    % ヤコビアンm_F
    m_F     = (result.Temp_xPred_x)' / (result.K + lamda_f * eye(N, N)) * result.X_t_1; % 式(10)
    PPred = m_F * PEst * m_F' + w;
    %% フィルタリングステップ
    % ヤコビアンm_Hのためのk*(x_t)の偏微分の計算
    for j = 1:N
        k = -beta_h*(X_t(j, 1) -  xPred)*alpha_h^2 * exp((-1 / 2) * beta_h * (X_t(j, 1) -  xPred)^2);
        result.Temp_xPred_y = [result.Temp_xPred_y; k];
    end
    % ヤコビアンm_H
    m_H     = (result.Temp_xPred_y)' / (result.K + lamda_h * eye(N, N)) * result.Y_t; % 式(12)
    %カルマンゲイン
    K = PPred * m_H' / (m_H * PPred * m_H' + v); % 式(18)
    % 状態推定値
    % m_hにxPredを入れた値を求めるためにk*(x_t)の計算
    for j = 1:N
        k = alpha_h^2 * exp((-1 / 2) * beta_h * (X_t(j, 1) -  xPred)^2);
        result.Temp_xEst_y = [result.Temp_xEst_y; k];
    end
    % 平均
    m_h =  (result.Temp_xEst_y)' / (result.K + lamda_h * eye(N, N)) * result.Y_t;
    xEst = xPred + K * (y - m_h); % 式(17)
    % 事後誤差共分散行列
    % ヤコビアンmfにxPredを入れた値を求めるためにk*(x_t)の偏微分の計算
    for j = 1:N
        k = -beta_f * (X_t(j, 1) -  xEst) * alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  xEst)^2);
        result.Temp_xEst_last = [result.Temp_xEst_last; k];
    end
    % ヤコビアンmfにxPredを入れた値を計算
    m_F  = (result.Temp_xEst_last)' / (result.K + lamda_f * eye(N, N)) * result.X_t_1; % 式(10)
    PEst = m_F^2 / (PPred + m_H^2 / v + theta) + w; % 式(19)
    % thetaの条件式
    if PPred + m_H^2 / v + theta <= 0
        break;
    end
    %% 配列に結果を格納
    result.x     = [result.x; x];
    result.y     = [result.y; y];
    result.time  = [result.time; time];
    result.xEst  = [result.xEst; xEst];
    result.m     = [result.m; m];
    result.sigma = [result.sigma; sigma];
    result.Upper = [result.Upper; Upper];
    result.Lower = [result.Lower; Lower];
    %% 一時保存用配列を空にしてforに戻る
    result.Temp           = [];
    result.Temp_x         = [];
    result.Temp_xEst      = [];
    result.Temp_xEst_y    = [];
    result.Temp_xEst_last = [];
    result.Temp_xPred_x   = [];
    result.Temp_xPred_y   = [];
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