%% ===================================================================== %%
%  Reconfigurable Intelligent Surface (RIS) Neural Network Model
%  Final validated solution based on physics-informed feature engineering.
%  Author: Yazhe Xu, University of California, Irvine
%  ---------------------------------------------------------------------
%  Objective:
%  Build a data-driven framework to predict RIS system sum-rate
%  using both linear regression and neural network models, and
%  evaluate algorithmic performance under realistic channel conditions.
%% ===================================================================== %%

clear; clc; close all;
rng(42);
%% === System Parameters ===
M = 8; N = 64; K = 4; d = 1;
sigma2 = 1e-3; P_BS = 10;

fprintf('=== RIS Final Solution ===\n');
fprintf('Physics-informed Sum Rate Prediction\n\n');

%% === Core Feature Extraction Function ===
% Extracts physics-based features from RIS channel and configuration.
function features = extract_physics_features(H, G, W, Theta, sigma2)
    K = length(G);
    Psi = diag(exp(1j * Theta));
    features = [];

    % (1) User-level SINR-related features
    for k = 1:K
        % Signal power for user k
        signal_power = abs(G{k} * Psi * H * W{k})^2;

        % Total interference power
        interference_power = 0;
        for j = 1:K
            if j ~= k
                interference_power = interference_power + abs(G{k} * Psi * H * W{j})^2;
            end
        end

        % SINR (linear & dB) and estimated rate
        sinr_linear = signal_power / (interference_power + sigma2 + 1e-12);
        sinr_db = 10*log10(sinr_linear + 1e-12);
        rate_estimate = log2(1 + sinr_linear);
        features = [features; signal_power; interference_power; sinr_linear; sinr_db; rate_estimate];
    end

    % (2) System-level power distribution metrics
    total_signal_power = 0;
    power_balance = 0;
    for k = 1:K
        signal_k = abs(G{k} * Psi * H * W{k})^2;
        total_signal_power = total_signal_power + signal_k;
        power_balance = power_balance + signal_k^2;
    end
    features = [features; total_signal_power; power_balance/total_signal_power^2];

    % (3) Channel condition metric (effective condition number)
    H_eff = Psi * H;
    cond_num = log10(cond(H_eff) + 1);
    features = [features; cond_num];
end

%% === Dataset Generation ===
N_samples = 800;
N_test = 200;

fprintf('Generating %d training samples...\n', N_samples);

% Determine feature dimension
H_temp = (randn(N, M) + 1j*randn(N, M))/sqrt(2);
G_temp = cell(K, 1);
W_temp = cell(K, 1);
for k = 1:K
    G_temp{k} = (randn(1, N) + 1j*randn(1, N))/sqrt(2);
    W_temp{k} = (randn(M, d) + 1j*randn(M, d))/sqrt(2);
end
Theta_temp = 2*pi*rand(N, 1);

features_temp = extract_physics_features(H_temp, G_temp, W_temp, Theta_temp, sigma2);
feature_dim = length(features_temp);
fprintf('Feature dimension: %d (vs. 1792 in raw vectorization)\n', feature_dim);

% Generate valid training samples
X_train = zeros(feature_dim, N_samples);
Y_train = zeros(1, N_samples);
valid_samples = 0;

for sample = 1:N_samples * 2  % Generate extra to filter invalid cases
    if valid_samples >= N_samples
        break;
    end

    % Randomized channel and beamforming generation
    H = (randn(N, M) + 1j*randn(N, M))/sqrt(2);
    G = cell(K, 1); 
    W = cell(K, 1);
    for k = 1:K
        G{k} = (randn(1, N) + 1j*randn(1, N))/sqrt(2);
        W{k} = (randn(M, d) + 1j*randn(M, d))/sqrt(2);
        W{k} = W{k} * sqrt(P_BS/K) / norm(W{k}, 'fro');
    end
    Theta = 2*pi*rand(N, 1);

    % Compute ground-truth sum-rate
    sum_rate = compute_sum_rate_stable(H, G, W, Theta, sigma2);

    % Validate range and store
    if isfinite(sum_rate) && sum_rate > 0 && sum_rate < 50
        valid_samples = valid_samples + 1;
        features = extract_physics_features(H, G, W, Theta, sigma2);
        X_train(:, valid_samples) = features;
        Y_train(valid_samples) = sum_rate;
        if mod(valid_samples, 100) == 0
            fprintf('Valid samples: %d/%d\n', valid_samples, N_samples);
        end
    end
end

X_train = X_train(:, 1:valid_samples);
Y_train = Y_train(1:valid_samples);
N_samples = valid_samples;

fprintf('Training data generation complete. Valid samples: %d\n', N_samples);
fprintf('Sum-rate range: %.4f – %.4f bps/Hz\n', min(Y_train), max(Y_train));

%% === Feature Correlation Analysis ===
fprintf('\nAnalyzing feature–target correlations...\n');

correlations = zeros(feature_dim, 1);
for i = 1:feature_dim
    if std(X_train(i, :)) > 1e-8
        corr_matrix = corrcoef(X_train(i, :), Y_train);
        correlations(i) = abs(corr_matrix(1, 2));
    end
end

fprintf('=== Feature Correlation Analysis ===\n');
fprintf('  Max correlation: %.4f\n', max(correlations));
fprintf('  Mean correlation: %.4f\n', mean(correlations));
fprintf('  Highly correlated features (>0.5): %d\n', sum(correlations > 0.5));
fprintf('  Moderately correlated features (>0.3): %d\n', sum(correlations > 0.3));

%% === Data Preprocessing ===
% Remove outliers and normalize features
Q1 = quantile(Y_train, 0.1);
Q3 = quantile(Y_train, 0.9);
valid_idx = (Y_train >= Q1) & (Y_train <= Q3);
X_train = X_train(:, valid_idx);
Y_train = Y_train(valid_idx);
N_samples = length(Y_train);

fprintf('\nAfter outlier removal: %d samples retained\n', N_samples);

X_mean = mean(X_train, 2);
X_std = std(X_train, 0, 2) + 1e-8;
X_train_norm = (X_train - X_mean) ./ X_std;
Y_mean = mean(Y_train);
Y_std = std(Y_train) + 1e-8;
Y_train_norm = (Y_train - Y_mean) / Y_std;

%% === Baseline Model: Linear Regression ===
fprintf('\nRunning baseline linear regression...\n');
[~, important_idx] = sort(correlations, 'descend');
good_features = find(correlations > 0.3);
n_top_features = min(12, length(good_features));
top_features = important_idx(1:n_top_features);
X_train_selected = X_train_norm(top_features, :);

if N_samples > n_top_features
    lambda_reg = 1e-6;
    A = X_train_selected * X_train_selected' + lambda_reg * eye(size(X_train_selected, 1));
    b = X_train_selected * Y_train_norm';
    beta = A \ b;
    Y_pred_linear_norm = beta' * X_train_selected;
    Y_pred_linear = Y_pred_linear_norm * Y_std + Y_mean;

    SS_res = sum((Y_train - Y_pred_linear).^2);
    SS_tot = sum((Y_train - Y_mean).^2);
    r2_linear = 1 - SS_res / SS_tot;

    fprintf('  Top %d features selected.\n', n_top_features);
    fprintf('  Linear Regression R²: %.6f\n', r2_linear);
else
    r2_linear = -1;
    fprintf('  Insufficient samples for linear regression.\n');
end

%% === Neural Network Training ===
fprintf('\nTraining neural network model...\n');
if r2_linear > 0.3 || r2_linear == -1
     hidden1 = 32;  
    hidden2 = 16; 
    learning_rate = 0.01;
    epochs = 500;

    % Initialize weights
     W1 = randn(hidden1, n_top_features) * sqrt(1/n_top_features);
    b1 = zeros(hidden1, 1);
    W2 = randn(hidden2, hidden1) * sqrt(1/hidden1);
    b2 = zeros(hidden2, 1);
    W3 = randn(1, hidden2) * sqrt(1/hidden2);
    b3 = 0;

   % Activation functions
    relu = @(x) max(0, x);
    relu_deriv = @(x) double(x > 0);

    loss_history = zeros(epochs, 1);
    fprintf('  Network structure: %d → %d → %d → 1\n', n_top_features, hidden1, hidden2);

    for epoch = 1:epochs
        % === Forward pass ===
        Z1 = W1 * X_train_selected + b1;
        A1 = relu(Z1);

        Z2 = W2 * A1 + b2;
        A2 = relu(Z2);

        Z3 = W3 * A2 + b3;
        Y_pred_norm = Z3;

        % Compute loss
        lambda = 1e-4;
        loss = mean((Y_pred_norm - Y_train_norm).^2) + ...
               lambda * (sum(W1(:).^2) + sum(W2(:).^2) + sum(W3(:).^2));
        loss_history(epoch) = loss;
        % Backpropagation
         dZ3 = 2 * (Y_pred_norm - Y_train_norm) / N_samples;
        dW3 = dZ3 * A2' + 2 * lambda * W3;
        db3 = sum(dZ3);
        dA2 = W3' * dZ3;

        dZ2 = dA2 .* relu_deriv(Z2);
        dW2 = dZ2 * A1' + 2 * lambda * W2;
        db2 = sum(dZ2, 2);
        dA1 = W2' * dZ2;

        dZ1 = dA1 .* relu_deriv(Z1);
        dW1 = dZ1 * X_train_selected' + 2 * lambda * W1;
        db1 = sum(dZ1, 2);

 % === Gradient clipping ===
        grad_clip = 1.0;
        dW1 = max(min(dW1, grad_clip), -grad_clip);
        dW2 = max(min(dW2, grad_clip), -grad_clip);
        dW3 = max(min(dW3, grad_clip), -grad_clip);

        % Update parameters
       W3 = W3 - learning_rate * dW3;
        b3 = b3 - learning_rate * db3;
        W2 = W2 - learning_rate * dW2;
        b2 = b2 - learning_rate * db2;
        W1 = W1 - learning_rate * dW1;
        b1 = b1 - learning_rate * db1;
    end

    % Evaluate on training set
    Z1 = W1 * X_train_selected + b1; A1 = relu(Z1);
    Z2 = W2 * A1 + b2; A2 = relu(Z2);
    Z3 = W3 * A2 + b3;
    Y_pred_train_norm = Z2;
    Y_pred_train = Y_pred_train_norm * Y_std + Y_mean;

    SS_res_train = sum((Y_train - Y_pred_train).^2);
    SS_tot_train = sum((Y_train - Y_mean).^2);
    r2_train = 1 - SS_res_train / SS_tot_train;

    fprintf('  Training complete. R² (train): %.6f\n', r2_train);
else
    fprintf('  Linear model insufficient, feature redesign recommended.\n');
    r2_train = r2_linear;
end

%% === Test Data Generation ===
fprintf('\n=== Generating test dataset... ===\n');

X_test = zeros(feature_dim, N_test);
Y_test = zeros(1, N_test);
valid_test = 0;

for sample = 1:N_test * 2
    if valid_test >= N_test
        break;
    end

    % Generate random test channel parameters
    H_test = (randn(N, M) + 1j*randn(N, M)) / sqrt(2);
    G_test = cell(K, 1);
    W_test = cell(K, 1);

    for k = 1:K
        G_test{k} = (randn(1, N) + 1j*randn(1, N)) / sqrt(2);
        W_test{k} = (randn(M, d) + 1j*randn(M, d)) / sqrt(2);
        W_test{k} = W_test{k} * sqrt(P_BS/K) / norm(W_test{k}, 'fro');
    end

    % Random phase configuration for RIS
    Theta_test = 2*pi*rand(N, 1);

    % Compute ground-truth sum-rate for test case
    sum_rate_test = compute_sum_rate_stable(H_test, G_test, W_test, Theta_test, sigma2);

    % Store only valid samples
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

fprintf('Test dataset generated successfully. Valid samples: %d\n', N_test);

%% === Model Evaluation on Test Set ===
fprintf('\n=== Evaluating model on test data... ===\n');

% Normalize test data using training statistics
X_test_norm = (X_test - X_mean) ./ X_std;
X_test_selected = X_test_norm(top_features, :);

if exist('W3', 'var')  
    % === Neural Network (2-layer) prediction ===
    Z1_test = W1 * X_test_selected + b1;
    A1_test = max(0, Z1_test);     % ReLU activation

    Z2_test = W2 * A1_test + b2;
    A2_test = max(0, Z2_test);     % Second ReLU

    Z3_test = W3 * A2_test + b3;
    Y_pred_test_norm = Z3_test;
    Y_pred_test = Y_pred_test_norm * Y_std + Y_mean;

    model_type = 'Neural Network (Optimized)';

elseif exist('W2', 'var')
    % === Neural Network (1-layer) prediction ===
    Z1_test = W1 * X_test_selected + b1;
    A1_test = tanh(Z1_test);
    Z2_test = W2 * A1_test + b2;
    Y_pred_test_norm = Z2_test;
    Y_pred_test = Y_pred_test_norm * Y_std + Y_mean;

    model_type = 'Neural Network (1-layer)';
else
    % === Linear regression baseline ===
    Y_pred_test_norm = beta' * X_test_selected;
    Y_pred_test = Y_pred_test_norm * Y_std + Y_mean;
    model_type = 'Linear Regression';
end

% === Performance Metrics ===
mse = mean((Y_pred_test - Y_test).^2);
rmse = sqrt(mse);
mae = mean(abs(Y_pred_test - Y_test));
SS_res_test = sum((Y_test - Y_pred_test).^2);
SS_tot_test = sum((Y_test - mean(Y_test)).^2);
r2_test = 1 - SS_res_test / SS_tot_test;

fprintf('Model type: %s\n', model_type);
fprintf('  MSE  = %.6f\n', mse);
fprintf('  RMSE = %.6f\n', rmse);
fprintf('  MAE  = %.6f\n', mae);
fprintf('  R²   = %.6f\n', r2_test);
fprintf('-------------------------------------------\n');

%% === Visualization ===
figure('Units','normalized','Position',[0.1 0.1 0.8 0.4]);

% (1) Feature Correlation
subplot(1, 3, 1);
bar(correlations, 'FaceColor', [0.2 0.6 0.8]);
xlabel('Feature Index', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Correlation Coefficient', 'FontSize', 12, 'FontWeight', 'bold');
title('Feature–Sum Rate Correlation', 'FontSize', 13);
grid on;

% (2) Training Loss Curve
if exist('loss_history', 'var')
    subplot(1, 3, 2);
    plot(1:epochs, loss_history, 'LineWidth', 2, 'Color', [0.1 0.4 0.8]);
    xlabel('Training Epochs', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Loss', 'FontSize', 12, 'FontWeight', 'bold');
    title('Training Loss Curve', 'FontSize', 13);
    grid on;
end

% (3) Predicted vs True
subplot(1, 3, 3);
scatter(Y_test, Y_pred_test, 50, 'b', 'filled');
hold on;
plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 'r--', 'LineWidth', 2);
xlabel('True Sum Rate');
ylabel('Predicted Sum Rate');
title(['Neural network model R² = ', num2str(r2_test, '%.4f')]);
grid on;
axis equal;


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