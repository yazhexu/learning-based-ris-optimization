%% ===================================================================== %%
%  Reconfigurable Intelligent Surface (RIS) Linear Regression Model
%  Final validated baseline solution using physics-informed features.
%  Author: Yazhe Xu, University of California, Irvine
%  ---------------------------------------------------------------------
%  Objective:
%  Build a physics-driven baseline model to predict RIS system sum-rate
%  using linear regression. The results will be compared later with
%  neural-network-based optimization approaches.
%% ===================================================================== %%

clear; clc; close all;
rng(42);
%% === System Parameters ===
M = 8; N = 64; K = 4; d = 1;
sigma2 = 1e-3; P_BS = 10;

fprintf('=== RIS Baseline Linear Regression ===\n');
fprintf('Physics-informed Sum Rate Prediction\n\n');
%% === Core Feature Extraction Function ===
function features = extract_physics_features(H, G, W, Theta, sigma2)
    K = length(G);
    Psi = diag(exp(1j * Theta));
    features = [];

    % (1) User-level SINR features
    for k = 1:K
        signal_power = abs(G{k} * Psi * H * W{k})^2;

        interference_power = 0;
        for j = 1:K
            if j ~= k
                interference_power = interference_power + abs(G{k} * Psi * H * W{j})^2;
            end
        end

        sinr_linear = signal_power / (interference_power + sigma2 + 1e-12);
        sinr_db = 10*log10(sinr_linear + 1e-12);
        rate_estimate = log2(1 + sinr_linear);

        features = [features; signal_power; interference_power; sinr_linear; sinr_db; rate_estimate];
    end

    % (2) System-level power and condition metrics
    total_signal_power = 0;
    power_balance = 0;
    for k = 1:K
        signal_k = abs(G{k} * Psi * H * W{k})^2;
        total_signal_power = total_signal_power + signal_k;
        power_balance = power_balance + signal_k^2;
    end
    features = [features; total_signal_power; power_balance/total_signal_power^2];

    % (3) Effective channel condition number
    H_eff = Psi * H;
    cond_num = log10(cond(H_eff) + 1);
    features = [features; cond_num];
end
%% === Data Generation ===
N_samples = 800;
N_test = 200;

fprintf('Generating %d training samples...\n', N_samples);

% Determine feature dimension
H_temp = (randn(N, M) + 1j*randn(N, M)) / sqrt(2);
G_temp = cell(K, 1);
W_temp = cell(K, 1);
for k = 1:K
    G_temp{k} = (randn(1, N) + 1j*randn(1, N)) / sqrt(2);
    W_temp{k} = (randn(M, d) + 1j*randn(M, d)) / sqrt(2);
end
Theta_temp = 2 * pi * rand(N, 1);

features_temp = extract_physics_features(H_temp, G_temp, W_temp, Theta_temp, sigma2);
feature_dim = length(features_temp);
fprintf('Feature dimension: %d (vs. 1792 in raw vectorization)\n', feature_dim);

% Initialize training dataset containers
X_train = zeros(feature_dim, N_samples);
Y_train = zeros(1, N_samples);
valid_samples = 0;

% Generate additional random samples to ensure enough valid data
for sample = 1:N_samples * 2
    if valid_samples >= N_samples
        break;
    end
    
    % Random channel and beamforming generation
    H = (randn(N, M) + 1j*randn(N, M)) / sqrt(2);
    G = cell(K, 1);
    W = cell(K, 1);
    for k = 1:K
        G{k} = (randn(1, N) + 1j*randn(1, N)) / sqrt(2);
        W{k} = (randn(M, d) + 1j*randn(M, d)) / sqrt(2);
        W{k} = W{k} * sqrt(P_BS / K) / norm(W{k}, 'fro');
    end
    
    Theta = 2 * pi * rand(N, 1);
    
    % Compute ground-truth system sum rate
    sum_rate = compute_sum_rate_stable(H, G, W, Theta, sigma2);
    
    % Accept only valid samples within a physical range
    if isfinite(sum_rate) && sum_rate > 0 && sum_rate < 50
        valid_samples = valid_samples + 1;
        features = extract_physics_features(H, G, W, Theta, sigma2);
        X_train(:, valid_samples) = features;
        Y_train(valid_samples) = sum_rate;
        
        if mod(valid_samples, 100) == 0
            fprintf('Valid samples: %d / %d\n', valid_samples, N_samples);
        end
    end
end

% Truncate arrays to valid sample count
X_train = X_train(:, 1:valid_samples);
Y_train = Y_train(1:valid_samples);
N_samples = valid_samples;

fprintf('Training data generation complete. Valid samples: %d\n', N_samples);
fprintf('Sum Rate range: %.4f – %.4f bps/Hz\n', min(Y_train), max(Y_train));
%% === Feature Analysis ===
fprintf('\nAnalyzing feature–target relationships...\n');

correlations = zeros(feature_dim, 1);
for i = 1:feature_dim
    if std(X_train(i, :)) > 1e-8
        corr_matrix = corrcoef(X_train(i, :), Y_train);
        correlations(i) = abs(corr_matrix(1, 2));
    end
end

fprintf('Feature correlation summary:\n');
fprintf('  Max correlation: %.4f\n', max(correlations));
fprintf('  Mean correlation: %.4f\n', mean(correlations));
fprintf('  Strong features (>0.5): %d\n', sum(correlations > 0.5));
fprintf('  Moderate features (>0.3): %d\n', sum(correlations > 0.3));

%% === Data Preprocessing ===
% Remove outliers (keep central 80% of samples)
Q1 = quantile(Y_train, 0.1);
Q3 = quantile(Y_train, 0.9);
valid_idx = (Y_train >= Q1) & (Y_train <= Q3);
X_train = X_train(:, valid_idx);
Y_train = Y_train(valid_idx);
N_samples = length(Y_train);

fprintf('\nData preprocessing completed.\n');
fprintf('  Remaining samples after outlier removal: %d\n', N_samples);

% Normalize features and target
X_mean = mean(X_train, 2);
X_std = std(X_train, 0, 2) + 1e-8;
X_train_norm = (X_train - X_mean) ./ X_std;

Y_mean = mean(Y_train);
Y_std = std(Y_train) + 1e-8;
Y_train_norm = (Y_train - Y_mean) / Y_std;

%% === Baseline Model: Linear Regression ===
fprintf('\n=== Baseline: Linear Regression ===\n');

% Select top correlated features
[~, important_idx] = sort(correlations, 'descend');
good_features = find(correlations > 0.3);
n_top_features = min(12, length(good_features));
top_features = important_idx(1:n_top_features);
X_train_selected = X_train_norm(top_features, :);

if N_samples > n_top_features
    % Regularized least squares (ridge regression)
    lambda_reg = 1e-6;
    A = X_train_selected * X_train_selected' + lambda_reg * eye(size(X_train_selected, 1));
    b = X_train_selected * Y_train_norm';
    beta = A \ b;
    Y_pred_linear_norm = beta' * X_train_selected;
    Y_pred_linear = Y_pred_linear_norm * Y_std + Y_mean;
    
    % Compute R²
    SS_res = sum((Y_train - Y_pred_linear).^2);
    SS_tot = sum((Y_train - Y_mean).^2);
    r2_linear = 1 - SS_res / SS_tot;
    
    fprintf('  Top %d correlated features selected.\n', n_top_features);
    fprintf('  Linear Regression R²: %.6f\n', r2_linear);
else
    r2_linear = -1;
    fprintf('  Insufficient samples for regression.\n');
end

%% === Test Data Generation ===
fprintf('\nGenerating test data...\n');

X_test = zeros(feature_dim, N_test);
Y_test = zeros(1, N_test);
valid_test = 0;

for sample = 1:N_test * 2
    if valid_test >= N_test
        break;
    end
    
    % Random test channels
    H_test = (randn(N, M) + 1j*randn(N, M)) / sqrt(2);
    G_test = cell(K, 1);
    W_test = cell(K, 1);
    for k = 1:K
        G_test{k} = (randn(1, N) + 1j*randn(1, N)) / sqrt(2);
        W_test{k} = (randn(M, d) + 1j*randn(M, d)) / sqrt(2);
        W_test{k} = W_test{k} * sqrt(P_BS / K) / norm(W_test{k}, 'fro');
    end
    Theta_test = 2 * pi * rand(N, 1);
    
    sum_rate_test = compute_sum_rate_stable(H_test, G_test, W_test, Theta_test, sigma2);
    
    if isfinite(sum_rate_test) && sum_rate_test > 0 && sum_rate_test < 50
        valid_test = valid_test + 1;
        features_test = extract_physics_features(H_test, G_test, W_test, Theta_test, sigma2);
        X_test(:, valid_test) = features_test;
        Y_test(valid_test) = sum_rate_test;
    end
end

X_test = X_test(:, 1:valid_test);
Y_test = Y_test(1:valid_test);
N_test = valid_test;
fprintf('Test data generation complete. Valid samples: %d\n', N_test);

%% === Model Evaluation on Test Set ===
X_test_norm = (X_test - X_mean) ./ X_std;
X_test_selected = X_test_norm(top_features, :);

Y_pred_test_norm = beta' * X_test_selected;
Y_pred_test = Y_pred_test_norm * Y_std + Y_mean;

mse = mean((Y_pred_test - Y_test).^2);
rmse = sqrt(mse);
mae = mean(abs(Y_pred_test - Y_test));
SS_res_test = sum((Y_test - Y_pred_test).^2);
SS_tot_test = sum((Y_test - mean(Y_test)).^2);
r2_test = 1 - SS_res_test / SS_tot_test;

fprintf('\n=== Test Performance ===\n');
fprintf('  MSE:  %.6f\n', mse);
fprintf('  RMSE: %.6f\n', rmse);
fprintf('  MAE:  %.6f\n', mae);
fprintf('  R²:   %.6f\n', r2_test);

%% === Visualization (2 Plots) ===
figure('Position', [100, 100, 1000, 400]);

% (1) Feature correlation bar chart
subplot(1,2,1);
bar(correlations);
xlabel('Feature Index');
ylabel('Correlation Coefficient');
title('Feature–Sum Rate Correlation');
grid on;

% (2) Predicted vs True Sum Rate
subplot(1,2,2);
scatter(Y_test, Y_pred_test, 40, 'b', 'filled');
hold on;
plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 'r--', 'LineWidth', 1.5);
xlabel('True Sum Rate');
ylabel('Predicted Sum Rate');
title(['Baseline model (R² = ', num2str(r2_test, '%.4f'), ')']);
axis equal;
grid on;

%% === Helper Function ===
function sum_rate = compute_sum_rate_stable(H, G, W, Theta, sigma2)
    K = length(G);
    try
        Psi = diag(exp(1j * Theta));
        A = cell(K, 1);
        for k = 1:K
            A{k} = G{k} * Psi * H * W{k};
        end
        gain = zeros(K, 1);
        for k = 1:K
            gain(k) = norm(A{k}, 'fro')^2;
        end
        [~, idx_asc] = sort(gain, 'ascend');
        rates = zeros(K, 1);
        for p = 1:K
            uid = idx_asc(p);
            signal_power = abs(A{uid})^2;
            interference = 0;
            for q = p+1:K
                if q <= K
                    iid = idx_asc(q);
                    interference = interference + abs(A{iid})^2;
                end
            end
            sinr = signal_power / (interference + sigma2 + 1e-12);
            sinr = min(max(sinr, 1e-6), 1e6);
            rates(uid) = log2(1 + sinr);
        end
        sum_rate = sum(rates);
        if ~isfinite(sum_rate) || sum_rate <= 0
            sum_rate = NaN;
        end
    catch
        sum_rate = NaN;
    end
end
