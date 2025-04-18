%% danish
da_single_lang = readmatrix("../telemetry_data/04_06_2025_single_language_telemetry_da");

train_loss_da = da_single_lang(1, :);
eval_loss_da = da_single_lang(2, :);

epochs = 1:length(train_loss_da);

figure;
plot(epochs, train_loss_da, '-o', 'LineWidth', 1.5);
hold on;
plot(epochs, eval_loss_da, '-s', 'LineWidth', 1.5);

xlabel('Epoch');
ylabel('Loss');
title('Training and Evaluation Loss of single decoder (DA)');
legend('Train Loss', 'Eval Loss');
grid on;

%% French

fr_single_lang = readmatrix("../telemetry_data/04_13_2020_single_language_telemetry_fr");

train_loss_fr = fr_single_lang(1:2:end, 1);
eval_loss_fr = fr_single_lang(2:2:end, 2);

epochs_f = 1:length(train_loss_fr);

figure;
plot(epochs_f, train_loss_fr, '-o', 'LineWidth', 1.5);
hold on;
plot(epochs_f, eval_loss_fr, '-s', 'LineWidth', 1.5);

xlabel('Epoch');
ylabel('Loss');
title('Training and Evaluation Loss of single decoder (FR)');
legend('Train Loss', 'Eval Loss');
grid on;


%% Multi Language

multi_lang = readmatrix("../telemetry_data/04_13_2025_multi_language_telemetry");

train_loss_da_m = multi_lang(1:4:end, 1);
train_loss_es_m = multi_lang(2:4:end, 2);
train_loss_fr_m = multi_lang(3:4:end, 3);

epochs_da = 1:3:3*length(train_loss_da_m);
epochs_es = 2:3:3*length(train_loss_es_m);
epochs_fr = 3:3:3*length(train_loss_es_m);


figure;
plot(epochs_da, train_loss_da_m, '-o', 'LineWidth', 1.5);
hold on;
plot(epochs_es, train_loss_es_m, '-o', 'LineWidth', 1.5);
hold on;
plot(epochs_fr, train_loss_fr_m, '-o', 'LineWidth', 1.5);


xlabel('Epoch');
ylabel('Loss');
title('Multi-Language Model train loss comparison');
legend('Danish Train Loss', 'Spanish Train Loss', 'Franch Train Loss');
grid on;

%eval loss

figure;
eval_loss_da_m = multi_lang(4:4:end, 4);
eval_loss_es_m = multi_lang(4:4:end, 5);
eval_loss_fr_m = multi_lang(4:4:end, 6);

epochs_eval = 3:3:3*length(eval_loss_da_m);
plot(epochs_eval, eval_loss_da_m, '-o', 'LineWidth', 1.5);
hold on;
plot(epochs_eval, eval_loss_es_m, '-o', 'LineWidth', 1.5);
hold on;
plot(epochs_eval, eval_loss_fr_m, '-o', 'LineWidth', 1.5);

xlabel('Epoch');
ylabel('Loss');
title('Multi-Language Model eval loss comparison');
legend('Danish Eval Loss', 'Spanish Eval Loss', 'Franch Eval Loss');
grid on;


%% Comparisons


% compare danish and french eval loss
figure();
plot(epochs, eval_loss_da, '-s', 'LineWidth', 1.5);
hold on;
plot(epochs_f, eval_loss_fr, '-s', 'LineWidth', 1.5);

xlabel('Epoch');
ylabel('Loss');
title('Comparison of Danish & French Single Decoder');
legend('Danish Eval Loss', 'Franch Eval Loss');
grid on;

%compare danish & french single decoder vs multi decoder
figure();
plot(epochs, eval_loss_da, '-s', 'LineWidth', 1.5);
hold on;
plot(epochs_f, eval_loss_fr, '-s', 'LineWidth', 1.5);
plot(epochs_eval, eval_loss_da_m, '-o', 'LineWidth', 1.5);
plot(epochs_eval, eval_loss_fr_m, '-o', 'LineWidth', 1.5);


xlabel('Epoch');
ylabel('Loss');
title('Comparison Danish & French single Vs multi decoder');
legend('Danish Single', 'Franch Single', 'Danish Multi', 'Franch Multi');
grid on;
xlim([0 30]);

%% Comparison Part II - fine tuning

%load all data
it_multi_decode = readmatrix("../telemetry_data/04_15_2025_finetune_from_multi_decoder_telemetry");
it_single_decode = readmatrix("../telemetry_data/04_15_2025_finetune_from_single_decoder_telemetry");
it_normal = readmatrix("../telemetry_data/04_15_2025_single_language_telemetry_it");

eval_it_multi_decode = it_multi_decode(:, 2);
eval_it_single_decode = it_single_decode(:, 2);
eval_it_normal = it_normal(2:2:end, 2);
eval_it_normal = eval_it_normal(1:1:30);

epochs_comp = 1:30;

plot(epochs_comp, eval_it_multi_decode, '-o', 'LineWidth', 1.5);
hold on;
plot(epochs_comp, eval_it_single_decode, '-o', 'LineWidth', 1.5);
hold on;
plot(epochs_comp, eval_it_normal, '-o', 'LineWidth', 1.5);

xlabel('Epoch');
ylabel('Evaluation Loss');
title('Comparison of Training W/WO Transfer Learning Methods');
legend('our method', 'transfer learning', 'baseline');
grid on;


%%
epochs_f = 1:length(eval_it_multi_decode);

figure;
plot(epochs_f, train_loss_fr, '-o', 'LineWidth', 1.5);
hold on;
plot(epochs_f, eval_loss_fr, '-s', 'LineWidth', 1.5);

xlabel('Epoch');
ylabel('Loss');
title('Training and Evaluation Loss of single decoder (FR)');
legend('Train Loss', 'Eval Loss');
grid on;



%% Save files

output_dir = fullfile(pwd, 'saved_figure');
print(gcf, 'loss_plot_vector', '-dsvg', '-r300');

% Get all figure handles
figs = findall(0, 'Type', 'figure');

% Save each figure as SVG
for i = 1:length(figs)
    fig = figs(i);
    figure(fig);  % Make it the current figure
    
    % Define output file name
    filename = fullfile(output_dir, sprintf('figure_%d.svg', i));
    
    % Save as SVG
    print(fig, filename, '-dsvg', '-r300');
end




