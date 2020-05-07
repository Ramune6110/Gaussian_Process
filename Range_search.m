clear;
close all;
clc;
%% �V�~�����[�V�������Ԑݒ�
dt      = 1;
time    = 0.0;
endtime = 50.0;
nsteps  = ceil((endtime - time) / dt);
%% ��Ԃ̏����l
x     = 10;
xEst  = 10;
PEst  = 1;
w     = 0.5^2;
v     = 0.05^2;
theta = -0.05;
%% �n�C�p�[�p�����[�^�̒l
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
%% �f�[�^�Z�b�g�擾�͈͋y�ь�
a = -20;
b = 20;
N = 500;
%% �t�B���^���ʊi�[
result.x    = [];
result.y    = [];
result.time = [];
result.xEst = [];
result.PEst = [];
%% �����l�ݒ�
result.x     = [result.x; x];
result.time  = [result.time; time];
result.xEst  = [result.xEst; xEst];
result.PEst  = [result.PEst; PEst];
%% �֐��̕��ϒl���i�[
result.m     = [];
%% �֐��̕��U���i�[
result.sigma = [];
result.Upper = [];
result.Lower = [];
%% �f�[�^�Z�b�g����
[X_t, X_t_1, Y_t, K, L] = makeDataset(a, b, w, v, alpha_f, alpha_h, beta_f, beta_h, N);
m = K(1,:) / (K + lamda_f * eye(N, N)) * X_t_1;
% ���U
sigma = alpha_f^2 - K(1,:) / (K + lamda_f * eye(N, N)) * (K(1,:))';
Upper = m + 3 * sqrt(sigma);
Lower = m - 3 * sqrt(sigma);
%% �f�[�^�Z�b�g���琄�肵���֐��̕��ςƕ��U�������l�Ƃ��Ĕz��Ɋi�[
result.m     = [result.m; m];
result.sigma = [result.sigma; sigma];
result.Upper = [result.Upper; Upper];
result.Lower = [result.Lower; Lower];
%% main loop
tic;% start�@
for i = 1:nsteps
    time = time + dt;
    % ----�K�E�X�ߒ���A�ɂ��֐�f�̐���-----
    % k*(x_t)�̌v�Z ��(4)
    k_star = Calc_k_star(N, alpha_f, beta_f, X_t, x);
    % ���� ��(8)
    m = (k_star)' / (K + lamda_f * eye(N, N)) * X_t_1;
    % ���U ��(8)
    sigma = alpha_f^2 - (k_star)' / (K + lamda_f * eye(N, N)) * k_star;
    Upper = m + 3 * sqrt(sigma);
    Lower = m - 3 * sqrt(sigma);
    %% �֐��l�̌v�Z
    x = 0.2 * x + ((25 * x) / (1 + x^2)) + 8 * cos(1.2 * x) +  sqrt(w) * randn(1, 1);
    y = sin(x / 10) + sqrt(v) * randn(1, 1);
    %---------------------------------------

    %% ------ ���X�N�s�q�^�t�B���^ -------
    %% �\���X�e�b�v
    % m_f��xEst����ꂽ�l���v�Z���邽�߂�k*(x_t)�̌v�Z
    k_star_xPred = Calc_k_star(N, alpha_f, beta_f, X_t, xEst);
    % m_f��xEst����ꂽ�l��m_xEst�Ƃ��Ă���
    m_xEst = (k_star_xPred)' / (K + lamda_f * eye(N, N)) * X_t_1;
    xPred  = m_xEst; % ��(16)
    % ���R�r�A��m_F�̂��߂�k*(x_t)�̕Δ����̌v�Z
    k_star_mf = Calc_k_star_mf(N, alpha_f, beta_f, X_t, xPred);
    % ���R�r�A��m_F
    m_F   = (k_star_mf)' / (K + lamda_f * eye(N, N)) * X_t_1; % ��(10)
    PPred = m_F * PEst * m_F' + w;
    %% �t�B���^�����O�X�e�b�v
    % ���R�r�A��m_H�̂��߂�k*(x_t)�̕Δ����̌v�Z
    k_star_mh = Calc_k_star_mf(N, alpha_h, beta_h, X_t, xPred);
    % ���R�r�A��m_H
    m_H = (k_star_mh)' / (K + lamda_h * eye(N, N)) * Y_t; % ��(12)
    %�J���}���Q�C��
    G = PPred * m_H' / (m_H * PPred * m_H' + v); % ��(18)
    % ��Ԑ���l
    % m_h��xPred����ꂽ�l�����߂邽�߂�k*(x_t)�̌v�Z
    k_star_xEst = Calc_k_star(N, alpha_h, beta_h, X_t, xPred);
    % ����
    m_h  = (k_star_xEst)' / (K + lamda_h * eye(N, N)) * Y_t;
    xEst = xPred + G * (y - m_h); % ��(17)
    % ����덷�����U�s��
    % ���R�r�A��mf��xPred����ꂽ�l�����߂邽�߂�k*(x_t)�̕Δ����̌v�Z
    k_star_xEst_last = Calc_k_star_mf(N, alpha_f, beta_f, X_t, xEst);
    % ���R�r�A��mf��xPred����ꂽ�l���v�Z
    m_F  = (k_star_xEst_last)' / (K + lamda_f * eye(N, N)) * X_t_1; % ��(10)
    PEst = m_F^2 / (PPred + m_H^2 / v + theta) + w; % ��(19)
    % theta�̏�����
    if PPred + m_H^2 / v + theta <= 0
        break;
    end
    %---------------------------------------

    %% �z��Ɍ��ʂ��i�[
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
    % �f�[�^�Z�b�g�Ə�ԕ����������X_t+1��Y_t�̃f�[�^�Z�b�g�𐶐�
    % ��ʂɁA��� (a,b) �� N �̗����́A�� r = a + (b-a).*rand(N,1) ���g���Đ����ł��܂��B
    X_t   = a + (b - a).*rand(N, 1);
    X_t_1 = zeros(N, 1); 
    Y_t   = zeros(N, 1);
    for i = 1:N
        X_t_1(i, 1) = 0.2 * X_t(i, 1) + ((25 * X_t(i, 1)) / (1 + X_t(i, 1)^2)) + 8 * cos(1.2 * X_t(i, 1)) +  sqrt(w) * randn(1, 1);
        Y_t(i, 1)   = sin(X_t(i, 1) / 10) + sqrt(v) * randn(1, 1);
    end
    % �f�[�^�Z�b�g����Ɋ֐�f�̕��ςƕ��U���v�Z
    K = zeros(N, N);
    for i = 1:N
        for j = 1:N
            K(i, j) = alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  X_t(i))^2);
        end
    end
    % �f�[�^�Z�b�g����Ɋ֐�h�̕��ςƕ��U���v�Z
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