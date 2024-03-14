% Clément Guichet, UGA CNRS UMR 5105 LPNC, Feb 2024

%% IMPORT PATH to DATA
clc
clearvars

addpath(genpath("H:\MEGAGING\code\Multilayer_LP\helper_functions\"));

% Load control mat file
rootdir = 'H:\MEGAGING\data\MEG\controle';
control.filelist = dir(fullfile(rootdir, '**\*.mat'));
control.filelist = struct2cell(control.filelist)';

% Load generation mat file
rootdir = 'H:\MEGAGING\data\MEG\generation';
generation.filelist = dir(fullfile(rootdir, '**\*.mat'));
generation.filelist = struct2cell(generation.filelist)';

%% Prune the connectivity matrices

for subj = 1:22
    % Control
    control.subjfile = strcat(control.filelist(subj,2), '\',control.filelist(subj,1));
    load(control.subjfile{1});
    control.subjmatrix(subj,:,:) = reshape(TF, [1, 1953, 6]);
    
    % Generation
    generation.subjfile = strcat(generation.filelist(subj,2), '\',generation.filelist(subj,1));
    load(generation.subjfile{1});
    generation.subjmatrix(subj,:,:) = reshape(TF, [1, 1953, 6]);

    for freq = 1:6
        % Control
        tmp = squeeze(control.subjmatrix(subj,:,:));
        control.freq.raw(subj,freq,:,:) = jVecToSymmetricMat(tmp(:,freq),62,0);
        
        tmp_freq = squeeze(control.freq.raw(subj,freq,:,:));
        [~, control.freq.pruned(subj,freq,:,:), ...
            control.freq.OMST_diagnostics.mdeg(subj,freq,:), ...
            control.freq.OMST_diagnostics.globalcosteffmax(subj,freq,:), ...
            control.freq.OMST_diagnostics.costmax(subj,freq,:),...
            control.freq.OMST_diagnostics.E(subj,freq,:)]...
            = threshold_omst_gce_wu(abs(tmp_freq),1);
        
         % Generation
        tmp = squeeze(generation.subjmatrix(subj,:,:));
        generation.freq.raw(subj,freq,:,:) = jVecToSymmetricMat(tmp(:,freq),62,0);
        
        tmp_freq = squeeze(generation.freq.raw(subj,freq,:,:));
        [~, generation.freq.pruned(subj,freq,:,:), ...
            generation.freq.OMST_diagnostics.mdeg(subj,freq,:), ...
            generation.freq.OMST_diagnostics.globalcosteffmax(subj,freq,:), ...
            generation.freq.OMST_diagnostics.costmax(subj,freq,:),...
            generation.freq.OMST_diagnostics.E(subj,freq,:)]...
            = threshold_omst_gce_wu(abs(tmp_freq),1);
        
        % OUTPUT: OMST-pruned subject-/frequency-specific connectivity matrix
        tmp_gen = squeeze(generation.freq.pruned(subj,freq,:,:));
        tmp_con = squeeze(control.freq.pruned(subj,freq,:,:));
        OUTPUT.generation(subj,freq,:,:) = tmp_gen;
        OUTPUT.control(subj,freq,:,:) = tmp_con;
        
        OUTPUT.density_gen(subj,freq,:) = ...
            nnz(...
            triu(...
            squeeze(OUTPUT.generation(subj,freq,:,:)...
            ))...
            )/(62*61/2)
        OUTPUT.density_control(subj,freq,:) = ...
            nnz(...
            triu(...
            squeeze(OUTPUT.control(subj,freq,:,:)...
            ))...
            )/(62*61/2)
    end
    
    OUTPUT.subjects_order = strrep(generation.filelist(:,2), 'H:\MEGAGING\data\MEG\generation\', '');
    OUTPUT.region = RowNames;
    
    clear control.subjfile generation.subjfile
end

clearvars -except control generation OUTPUT

%% Specify each layer
OUTPUT.generation_delta = squeeze(OUTPUT.generation(:,1,:,:));
OUTPUT.generation_theta = squeeze(OUTPUT.generation(:,2,:,:));
OUTPUT.generation_alpha = squeeze(OUTPUT.generation(:,3,:,:));
OUTPUT.generation_beta = squeeze(OUTPUT.generation(:,4,:,:));
OUTPUT.generation_gamma1 = squeeze(OUTPUT.generation(:,5,:,:));
OUTPUT.generation_gamma2 = squeeze(OUTPUT.generation(:,6,:,:));

OUTPUT.control_delta = squeeze(OUTPUT.control(:,1,:,:));
OUTPUT.control_theta = squeeze(OUTPUT.control(:,2,:,:));
OUTPUT.control_alpha = squeeze(OUTPUT.control(:,3,:,:));
OUTPUT.control_beta = squeeze(OUTPUT.control(:,4,:,:));
OUTPUT.control_gamma1 = squeeze(OUTPUT.control(:,5,:,:));
OUTPUT.control_gamma2 = squeeze(OUTPUT.control(:,6,:,:));

save('../../output/main_output.mat','OUTPUT')
%% Construct supra-adjacency matrices 

nrois = 62; % specify # of regions
nlrs = 6; % specify # of layers
id = nrois:nrois:nrois*nlrs;
% Pre-allocate
supra_generation = zeros(nrois*nlrs, nrois*nlrs, 22);
supra_control = zeros(nrois*nlrs, nrois*nlrs, 22);

% Construct
for subj = 1:22
    fprintf(1, 'Now constructing supra generation for sub %s!\n', num2str(subj))
    
    supratmp_gen = blkdiag(...
        squeeze(OUTPUT.generation_delta(subj,:,:)), ...
        squeeze(OUTPUT.generation_theta(subj,:,:)), ...
        squeeze(OUTPUT.generation_alpha(subj,:,:)), ...
        squeeze(OUTPUT.generation_beta(subj,:,:)), ...
        squeeze(OUTPUT.generation_gamma1(subj,:,:)), ...
        squeeze(OUTPUT.generation_gamma2(subj,:,:))...
        );
    for i = 1:length(id)
        supratmp_gen(id(i)*size(supratmp_gen,1)+1:size(supratmp_gen,1)+1:end) = 1;
        supratmp_gen(id(i)+1:size(supratmp_gen, 1)+1:1+size(supratmp_gen, 1)*min(size(supratmp_gen, 1)-id(i),size(supratmp_gen, 2))) = 1;
    end
    supra_generation(:,:,subj) = supratmp_gen;
    
    
    fprintf(1, 'Now constructing supra control for sub %s!\n', num2str(subj))
    
    supratmp_con = blkdiag(...
        squeeze(OUTPUT.control_delta(subj,:,:)), ...
        squeeze(OUTPUT.control_theta(subj,:,:)), ...
        squeeze(OUTPUT.control_alpha(subj,:,:)), ...
        squeeze(OUTPUT.control_beta(subj,:,:)), ...
        squeeze(OUTPUT.control_gamma1(subj,:,:)), ...
        squeeze(OUTPUT.control_gamma2(subj,:,:))...
        );
    for i = 1:length(id)
        supratmp_con(id(i)*size(supratmp_con,1)+1:size(supratmp_con,1)+1:end) = 1;
        supratmp_con(id(i)+1:size(supratmp_con, 1)+1:1+size(supratmp_con, 1)*min(size(supratmp_con, 1)-id(i),size(supratmp_con, 2))) = 1;
    end
    supra_control(:,:,subj) = supratmp_con;
end


save('../../output/supra_generation.mat','supra_generation')
save('../../output/supra_control.mat','supra_control')

%% Extract the monolayer eigenvector centrality

% Calculate nodal EC
for subj = 1:22
    fprintf(1, 'Now computing EC for sub %s!\n', num2str(subj))
    
    OUTPUT.EC.control.delta(subj,:) = eigenvector_centrality_und(squeeze(OUTPUT.control_delta(subj,:,:)));
    OUTPUT.EC.control.theta(subj,:) = eigenvector_centrality_und(squeeze(OUTPUT.control_theta(subj,:,:)));
    OUTPUT.EC.control.alpha(subj,:) = eigenvector_centrality_und(squeeze(OUTPUT.control_alpha(subj,:,:)));
    OUTPUT.EC.control.beta(subj,:) = eigenvector_centrality_und(squeeze(OUTPUT.control_beta(subj,:,:)));
    OUTPUT.EC.control.gamma1(subj,:) = eigenvector_centrality_und(squeeze(OUTPUT.control_gamma1(subj,:,:)));
    OUTPUT.EC.control.gamma2(subj,:) = eigenvector_centrality_und(squeeze(OUTPUT.control_gamma2(subj,:,:)));
    
    OUTPUT.EC.generation.delta(subj,:) = eigenvector_centrality_und(squeeze(OUTPUT.generation_delta(subj,:,:)));
    OUTPUT.EC.generation.theta(subj,:) = eigenvector_centrality_und(squeeze(OUTPUT.generation_theta(subj,:,:)));
    OUTPUT.EC.generation.alpha(subj,:) = eigenvector_centrality_und(squeeze(OUTPUT.generation_alpha(subj,:,:)));
    OUTPUT.EC.generation.beta(subj,:) = eigenvector_centrality_und(squeeze(OUTPUT.generation_beta(subj,:,:)));
    OUTPUT.EC.generation.gamma1(subj,:) = eigenvector_centrality_und(squeeze(OUTPUT.generation_gamma1(subj,:,:)));
    OUTPUT.EC.generation.gamma2(subj,:) = eigenvector_centrality_und(squeeze(OUTPUT.generation_gamma2(subj,:,:)));
end

monolayer_EC_control = OUTPUT.EC.control;
save('../../output/eigenvector_centrality_monolayer_control.mat','monolayer_EC_control')
monolayer_EC_generation = OUTPUT.EC.generation;
save('../../output/eigenvector_centrality_monolayer_generation.mat','monolayer_EC_generation')