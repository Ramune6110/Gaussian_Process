clear;
close all;
clc;
%% �V�~�����[�V�������Ԑݒ�
dt      = 1;
time    = 0.0;
endtime = 50.0;
nsteps  = ceil((endtime - time) / dt);
%% 
% �t�B���^���ʊi�[
result.x    = [];
result.y    = [];
result.time = [];
result.xEst = [];
result.PEst = [];
% �����K�E�X�ߒ���A�̍s��K�ƈꎞ�ۑ��p�z��temp
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
% �f�[�^�Z�b�g
% ��ʂɁA��� (a,b) �� N �̗����́A�� r = a + (b-a).*rand(N,1) ���g���Đ����ł��܂��B
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
% ����
m = result.K(1,:) / (result.K + lamda_f * eye(N, N)) * result.X_t_1;
% ���U
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
tic;% start�@for����鎞�Ԃ̌v��
for i = 1:nsteps
    time = time + dt;
    % k*(x_t)�̌v�Z
    for j = 1:N
        k = alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  x)^2);
        result.Temp_x = [result.Temp_x; k];
    end
    % ����
    m =  (result.Temp_x)' / (result.K + lamda_f * eye(N, N)) * result.X_t_1;
    % ���U
    sigma = alpha_f^2 - (result.Temp_x)' / (result.K + lamda_f * eye(N, N)) * result.Temp_x;
    Upper = m + 3 * sqrt(sigma);
    Lower = m - 3 * sqrt(sigma);
    if time >= 25 && time <= 30 
        % �֐��l�̌v�Z
        x = 0.2 * x + ((25 * x) / (1 + x^2)) + 8 * cos(1.2 * x) +  10 * sqrt(w) * randn(1, 1);
        y = sin(x / 10) + 10 * sqrt(v) * randn(1, 1);
    else
        % �֐��l�̌v�Z
        x = 0.2 * x + ((25 * x) / (1 + x^2)) + 8 * cos(1.2 * x) +  sqrt(w) * randn(1, 1);
        y = sin(x / 10) + sqrt(v) * randn(1, 1);
    end
    % ------���X�N�s�q�^�t�B���^ --------
    % �\���X�e�b�v
    % k*(x_t)�̌v�Z
    for j = 1:N
        k = alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  xEst)^2);
        result.Temp_xEst = [result.Temp_xEst; k];
    end
    % ����
    m_xEst =  (result.Temp_xEst)' / (result.K + lamda_f * eye(N, N)) * result.X_t_1;
    xPred = m_xEst; % ��(16)
    % k*(x_t)�̕Δ����̌v�Z
    for j = 1:N
        k = -beta_f * (X_t(j, 1) -  xPred) * alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  xPred)^2);
        result.Temp_xPred_x = [result.Temp_xPred_x; k];
    end
    % ���R�r�A��F(m_F)
    F     = (result.Temp_xPred_x)' / (result.K + lamda_f * eye(N, N)) * result.X_t_1; % ��(10)
    PPred = F * PEst * F' + w;

    % �t�B���^�����O�X�e�b�v
    % k*(x_t)�̕Δ����̌v�Z
    for j = 1:N
        k = -beta_h*(X_t(j, 1) -  xPred)*alpha_h^2 * exp((-1 / 2) * beta_h * (X_t(j, 1) -  xPred)^2);
        result.Temp_xPred_y = [result.Temp_xPred_y; k];
    end
    % ���R�r�A��H(m_H)
    H     = (result.Temp_xPred_y)' / (result.K + lamda_h * eye(N, N)) * result.Y_t; % ��(12)
    %�J���}���Q�C��
    K = PPred*H' / (H*PPred*H' + v); % ��(18)
    %��Ԑ���l
    % k*(x_t)�̌v�Z
    for j = 1:N
        k = alpha_h^2 * exp((-1 / 2) * beta_h * (X_t(j, 1) -  xPred)^2);
        result.Temp_xEst_y = [result.Temp_xEst_y; k];
    end
    % ����
    m_h =  (result.Temp_xEst_y)' / (result.K + lamda_h * eye(N, N)) * result.Y_t;
    xEst = xPred + K * (y - m_h); % ��(17)
    %����덷�����U�s��
    % k*(x_t)�̕Δ����̌v�Z
    for j = 1:N
        k = -beta_f * (X_t(j, 1) -  xEst) * alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  xEst)^2);
        result.Temp_xEst_last = [result.Temp_xEst_last; k];
    end
    
    % ���R�r�A��F
    F     = (result.Temp_xEst_last)' / (result.K + lamda_f * eye(N, N)) * result.X_t_1; % ��(10)
    theta = 0.05;
    PEst  = F^2 / (PPred + H^2 / v + theta) + w; % ��(19)



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