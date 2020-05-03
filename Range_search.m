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
theta = 0.01;
%% �n�C�p�[�p�����[�^�̒l
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
% �����l�ݒ�
result.x     = [result.x; x];
result.time  = [result.time; time];
result.xEst  = [result.xEst; xEst];
result.PEst  = [result.PEst; PEst];
%% �����K�E�X�ߒ���A�̍s��K�ƈꎞ�ۑ��p�z��temp
result.K    = [];
result.temp = [];
%% �f�[�^�Z�b�g�i�[�z��
result.X_t_1 = [];
result.Y_t   = [];
%% m_f,m_h,m_F,m_H�̌v�Z�ߒ��Ŏg�p����ꎞ�ۑ��p�z��
result.Temp_x         = [];
result.Temp_xEst      = [];
result.Temp_xEst_y    = [];
result.Temp_xEst_last = [];
result.Temp_xPred_x   = [];
result.Temp_xPred_y   = [];
%% �֐��̕��ϒl���i�[
result.m      = [];
result.m_xEst = [];
%% �֐��̕��U���i�[
result.sigma = [];
result.Upper = [];
result.Lower = [];
%% �f�[�^�Z�b�g�Ə�ԕ����������X_t+1��Y_t�̃f�[�^�Z�b�g�𐶐�
% ��ʂɁA��� (a,b) �� N �̗����́A�� r = a + (b-a).*rand(N,1) ���g���Đ����ł��܂��B
X_t = a + (b - a).*rand(N, 1);
for i = 1:N
    X_t_1 = 0.2 * X_t(i, 1) + ((25 * X_t(i, 1)) / (1 + X_t(i, 1)^2)) + 8 * cos(1.2 * X_t(i, 1)) +  sqrt(w) * randn(1, 1);
    Y_t   = sin(X_t(i, 1) / 10) + sqrt(v) * randn(1, 1);
    result.X_t_1 = [result.X_t_1; X_t_1];
    result.Y_t   = [result.Y_t; Y_t];
end
%% �f�[�^�Z�b�g����Ɋ֐�f�̕��ςƕ��U���v�Z
for i = 1:N
    for j = 1:N
        k = alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  X_t(i))^2);
        result.temp = [result.temp; k];
    end
    result.K = vertcat(result.K, result.temp');
    result.temp = [];
end
% ����
m = result.K(1,:) / (result.K + lamda_f * eye(N, N)) * result.X_t_1;
% ���U
sigma = alpha_f^2 - result.K(1,:) / (result.K + lamda_f * eye(N, N)) * (result.K(1,:))';
Upper = m + sqrt(sigma);
Lower = m - sqrt(sigma);
%% �f�[�^�Z�b�g���琄�肵���֐��̕��ςƕ��U�������l�Ƃ��Ĕz��Ɋi�[
result.m      = [result.m; m];
result.m_xEst = [result.m_xEst; m];
result.sigma  = [result.sigma; sigma];
result.Upper  = [result.Upper; Upper];
result.Lower  = [result.Lower; Lower];
%% main loop
tic;% start�@
for i = 1:nsteps
    time = time + dt;
    %% �K�E�X�ߒ���A�ɂ��֐�f�̐���
    % k*(x_t)�̌v�Z ��(4)
    for j = 1:N
        k = alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  x)^2);
        result.Temp_x = [result.Temp_x; k];
    end
    % ���� ��(8)
    m =  (result.Temp_x)' / (result.K + lamda_f * eye(N, N)) * result.X_t_1;
    % ���U ��(8)
    sigma = alpha_f^2 - (result.Temp_x)' / (result.K + lamda_f * eye(N, N)) * result.Temp_x;
    Upper = m + 3 * sqrt(sigma);
    Lower = m - 3 * sqrt(sigma);
    %% �֐��l�̌v�Z
    x = 0.2 * x + ((25 * x) / (1 + x^2)) + 8 * cos(1.2 * x) +  sqrt(w) * randn(1, 1);
    y = sin(x / 10) + sqrt(v) * randn(1, 1);
    %% ------���X�N�s�q�^�t�B���^ --------
    %% �\���X�e�b�v
    % m_f��xEst����ꂽ�l���v�Z���邽�߂�k*(x_t)�̌v�Z
    for j = 1:N
        k = alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  xEst)^2);
        result.Temp_xEst = [result.Temp_xEst; k];
    end
    % m_f��xEst����ꂽ�l��m_xEst�Ƃ��Ă���
    m_xEst =  (result.Temp_xEst)' / (result.K + lamda_f * eye(N, N)) * result.X_t_1;
    xPred = m_xEst; % ��(16)
    % ���R�r�A��m_F�̂��߂�k*(x_t)�̕Δ����̌v�Z
    for j = 1:N
        k = -beta_f * (X_t(j, 1) -  xPred) * alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  xPred)^2);
        result.Temp_xPred_x = [result.Temp_xPred_x; k];
    end
    % ���R�r�A��m_F
    m_F     = (result.Temp_xPred_x)' / (result.K + lamda_f * eye(N, N)) * result.X_t_1; % ��(10)
    PPred = m_F * PEst * m_F' + w;
    %% �t�B���^�����O�X�e�b�v
    % ���R�r�A��m_H�̂��߂�k*(x_t)�̕Δ����̌v�Z
    for j = 1:N
        k = -beta_h*(X_t(j, 1) -  xPred)*alpha_h^2 * exp((-1 / 2) * beta_h * (X_t(j, 1) -  xPred)^2);
        result.Temp_xPred_y = [result.Temp_xPred_y; k];
    end
    % ���R�r�A��m_H
    m_H     = (result.Temp_xPred_y)' / (result.K + lamda_h * eye(N, N)) * result.Y_t; % ��(12)
    %�J���}���Q�C��
    K = PPred * m_H' / (m_H * PPred * m_H' + v); % ��(18)
    % ��Ԑ���l
    % m_h��xPred����ꂽ�l�����߂邽�߂�k*(x_t)�̌v�Z
    for j = 1:N
        k = alpha_h^2 * exp((-1 / 2) * beta_h * (X_t(j, 1) -  xPred)^2);
        result.Temp_xEst_y = [result.Temp_xEst_y; k];
    end
    % ����
    m_h =  (result.Temp_xEst_y)' / (result.K + lamda_h * eye(N, N)) * result.Y_t;
    xEst = xPred + K * (y - m_h); % ��(17)
    % ����덷�����U�s��
    % ���R�r�A��mf��xPred����ꂽ�l�����߂邽�߂�k*(x_t)�̕Δ����̌v�Z
    for j = 1:N
        k = -beta_f * (X_t(j, 1) -  xEst) * alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  xEst)^2);
        result.Temp_xEst_last = [result.Temp_xEst_last; k];
    end
    % ���R�r�A��mf��xPred����ꂽ�l���v�Z
    m_F  = (result.Temp_xEst_last)' / (result.K + lamda_f * eye(N, N)) * result.X_t_1; % ��(10)
    PEst = m_F^2 / (PPred + m_H^2 / v + theta) + w; % ��(19)
    % theta�̏�����
    if PPred + m_H^2 / v + theta <= 0
        break;
    end
    %% �z��Ɍ��ʂ��i�[
    result.x     = [result.x; x];
    result.y     = [result.y; y];
    result.time  = [result.time; time];
    result.xEst  = [result.xEst; xEst];
    result.m     = [result.m; m];
    result.sigma = [result.sigma; sigma];
    result.Upper = [result.Upper; Upper];
    result.Lower = [result.Lower; Lower];
    %% �ꎞ�ۑ��p�z�����ɂ���for�ɖ߂�
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