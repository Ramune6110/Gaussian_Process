clear;
close all;
clc;
%% シミュレーション時間設定
dt      = 0.1;
time    = 0.0;
endtime = 50.0;
nsteps  = ceil((endtime - time) / dt);
%% 状態の初期値
x     = 10;
xEst  = 10;
PEst  = 1;
w     = 0.5^2;
v     = 0.05^2;
theta = 0;
%% データセット取得範囲及び個数
a = -20;
b = 20;
N = 500;
%% ハイパーパラメータの初期値
alpha_f = 1.0;
beta_f  = 4.2;
lamda_f = 0.5;
alpha_h = 1.0;
beta_h  = 0.2;
lamda_h = 0.05;
%% 初期ガウス過程回帰の行列Kと一時保存用配列temp
result.K = [];
result.L = [];
result.temp = [];
%% データセット格納配列
result.X_t_1 = [];
result.Y_t   = [];
%% 勾配用配列
result.grad_alpha_f = [];
result.grad_beta_f = [];
result.Cf_alpha = [];
result.Cf_beta = [];
result.grad_alpha_h = [];
result.grad_beta_h = [];
result.Ch_alpha = [];
result.Ch_beta = [];
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
%% データセットを基に関数hの平均と分散を計算
for i = 1:N
    for j = 1:N
        k = alpha_h^2 * exp((-1 / 2) * beta_h * (X_t(j, 1) -  X_t(i))^2);
        result.temp = [result.temp; k];
    end
    result.L = vertcat(result.L, result.temp');
    result.temp = [];
end

%% main loop
tic;% start　
for i = 1:nsteps
    time = time + dt;
    % grad_alpha_f
    for i = 1:N
        for j = 1:N
            k = 2 * alpha_f * exp((-1 / 2) * beta_f * (X_t(j, 1) -  X_t(i))^2);
            result.Cf_alpha = [result.Cf_alpha; k];
        end
        result.grad_alpha_f = vertcat(result.grad_alpha_f, result.Cf_alpha');
        result.Cf_alpha = [];
    end
    % grad_beta_f
    for i = 1:N
        for j = 1:N
            k = (- 1 / 2) * (X_t(j, 1) -  X_t(i)) *  alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  X_t(i))^2);
            result.Cf_beta = [result.Cf_beta; k];
        end
        result.grad_beta_f = vertcat(result.grad_beta_f, result.Cf_beta');
        result.Cf_beta = [];
    end
    % grad_alpha_h
    for i = 1:N
        for j = 1:N
            k = 2 * alpha_h * exp((-1 / 2) * beta_h * (X_t(j, 1) -  X_t(i))^2);
            result.Ch_alpha = [result.Ch_alpha; k];
        end
        result.grad_alpha_h = vertcat(result.grad_alpha_h, result.Ch_alpha');
        result.Ch_alpha = [];
    end
    % grad_beta_h
    for i = 1:N
        for j = 1:N
            k = (- 1 / 2) * (X_t(j, 1) -  X_t(i)) *  alpha_h^2 * exp((-1 / 2) * beta_h * (X_t(j, 1) -  X_t(i))^2);
            result.Ch_beta = [result.Ch_beta; k];
        end
        result.grad_beta_h = vertcat(result.grad_beta_h, result.Ch_beta');
        result.Ch_beta = [];
    end

    Cf = result.K + lamda_f * eye(N, N);
    Ch = result.L + lamda_h * eye(N, N);
    % alpha_fの更新
    grad_alpha_f = (-1 / 2) * trace(inv(Cf) * result.grad_alpha_f) + (1 / 2) * (result.X_t_1)' / Cf * result.grad_alpha_f / Cf * result.X_t_1;
    alpha_f = alpha_f + dt * grad_alpha_f;
    % beta_fの更新
    grad_beta_f = (-1 / 2) * trace(inv(Cf) * result.grad_beta_f) + (1 / 2) * (result.X_t_1)' / Cf * result.grad_beta_f / Cf * result.X_t_1;
    beta_f = beta_f + dt * grad_beta_f;
    % lamda_fの更新
    grad_lamda_f = (-1 / 2) * trace(inv(Cf) * eye(N, N)) + (1 / 2) * (result.X_t_1)' / Cf * eye(N, N) / Cf * result.X_t_1;
    lamda_f = lamda_f + dt * grad_lamda_f;
    % alpha_hの更新
    grad_alpha_h = (-1 / 2) * trace(inv(Ch) * result.grad_alpha_h) + (1 / 2) * (result.X_t_1)' / Ch * result.grad_alpha_h / Ch * result.X_t_1;
    alpha_h = alpha_h + dt * grad_alpha_h;
    % beta_hの更新
    grad_beta_h = (-1 / 2) * trace(inv(Ch) * result.grad_beta_h) + (1 / 2) * (result.X_t_1)' / Ch * result.grad_beta_h / Ch * result.X_t_1;
    beta_h = beta_h + dt * grad_beta_h;
    % lamda_hの更新
    grad_lamda_h = (-1 / 2) * trace(inv(Ch) * eye(N, N)) + (1 / 2) * (result.X_t_1)' / Ch * eye(N, N) / Ch * result.X_t_1;
    lamda_h = lamda_h + dt * grad_lamda_h;
    % Reset
    result.grad_alpha_f = [];
    result.grad_beta_f = [];
    result.grad_alpha_h = [];
    result.grad_beta_h = [];
end
toc;% end
