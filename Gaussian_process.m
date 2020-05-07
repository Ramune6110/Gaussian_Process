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
theta = 0.05;
%% �n�C�p�[�p�����[�^�̒l
alpha_f = 1.937;
beta_f  = 0.400;
lamda_f = 4.126;
alpha_h = 0.999;
beta_h  = 0.002;
lamda_h = 0.198;
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
%% �֐��̕��ϒl���i�[
result.m      = [];
%% �֐��̕��U���i�[
result.sigma = [];
result.Upper = [];
result.Lower = [];
%% �f�[�^�Z�b�g�Ə�ԕ����������X_t+1��Y_t�̃f�[�^�Z�b�g�𐶐�
% ��ʂɁA��� (a,b) �� N �̗����́A�� r = a + (b-a).*rand(N,1) ���g���Đ����ł��܂��B
X_t   = a + (b - a).*rand(N, 1);
X_t_1 = zeros(N, 1); 
Y_t   = zeros(N, 1);
for i = 1:N
    X_t_1(i, 1) = 0.2 * X_t(i, 1) + ((25 * X_t(i, 1)) / (1 + X_t(i, 1)^2)) + 8 * cos(1.2 * X_t(i, 1)) +  sqrt(w) * randn(1, 1);
    Y_t(i, 1)   = sin(X_t(i, 1) / 10) + sqrt(v) * randn(1, 1);
end
%% �f�[�^�Z�b�g����Ɋ֐�f�̕��ςƕ��U���v�Z
K = zeros(N, N);
for i = 1:N
    for j = 1:N
        K(i, j) = alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  X_t(i))^2);
    end
end
%% �f�[�^�Z�b�g����Ɋ֐�h�̕��ςƕ��U���v�Z
L = zeros(N, N);
for i = 1:N
    for j = 1:N
        L(i, j) = alpha_h^2 * exp((-1 / 2) * beta_h * (X_t(j, 1) -  X_t(i))^2);
    end
end
m = K(1,:) / (K + lamda_f * eye(N, N)) * X_t_1;
% ���U
sigma = alpha_f^2 - K(1,:) / (K + lamda_f * eye(N, N)) * (K(1,:))';
Upper = m + 3 * sqrt(sigma);
Lower = m - 3 * sqrt(sigma);
%% �f�[�^�Z�b�g���琄�肵���֐��̕��ςƕ��U�������l�Ƃ��Ĕz��Ɋi�[
result.m      = [result.m; m];
result.sigma  = [result.sigma; sigma];
result.Upper  = [result.Upper; Upper];
result.Lower  = [result.Lower; Lower];
%% main loop
tic;% start�@
k_star = zeros(N, 1);
for i = 1:nsteps
    time = time + dt;
    %% �K�E�X�ߒ���A�ɂ��֐�f�̐���
    % k*(x_t)�̌v�Z ��(4)
    for j = 1:N
        k_star(j, 1) = alpha_f^2 * exp((-1 / 2) * beta_f * (X_t(j, 1) -  x)^2);
    end
    % ���� ��(8)
    m =  (k_star)' / (K + lamda_f * eye(N, N)) * X_t_1;
    % ���U ��(8)
    sigma = alpha_f^2 - (k_star)' / (K + lamda_f * eye(N, N)) * k_star;
    Upper = m + 3 * sqrt(sigma);
    Lower = m - 3 * sqrt(sigma);
    %% �֐��l�̌v�Z
    x = 0.2 * x + ((25 * x) / (1 + x^2)) + 8 * cos(1.2 * x) +  sqrt(w) * randn(1, 1);
    y = sin(x / 10) + sqrt(v) * randn(1, 1);
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

function [] = Drow(result)
    figure(1);
    plot(result.time, result.x, 'k'); hold on;
    plot(result.time, result.Upper, 'r');hold on;
    plot(result.time, result.m, 'g'); hold on;
    plot(result.time, result.Lower, 'b');hold on;
end