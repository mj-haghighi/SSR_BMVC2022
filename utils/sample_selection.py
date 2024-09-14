import torch
from utils.funcs import * 
from utils.relabeling import relabel_sample


def select_extended_samples(
    feature_bank, human_labels, modified_labels, args, knn_k, 
    human_labels_score_window, relabeled_human_labels_score_window, sample_pred_label_window,
    logger):

    if args.extended_sampleing_strategy == ExtendedSampleingStrategy.RELABELD_CONFIDENCE:
        return select_sample_based_on_modified_labales_confidence(feature_bank, modified_labels, args, knn_k, relabeled_human_labels_score_window, logger)
    elif args.relabeling_strategy == RelabelingStrategy.SAMPLE_STABLE
        return select_sample_based_on_sample_stable(feature_bank, modified_labels, args, knn_k, sample_pred_label_window, logger):
    
    raise Exception("sampling strategy is not exist")

def select_sample_based_on_modified_labales_confidence(feature_bank, labels, args, knn_k, relabeled_human_labels_score_window, logger):
    prediction_knn = weighted_knn(feature_bank, feature_bank, labels, args.num_classes, knn_k, 10)  # temperature in weighted KNN
    vote_y = torch.gather(prediction_knn, 1, labels.view(-1, 1)).squeeze()
    vote_max = prediction_knn.max(dim=1)[0]
    right_score = vote_y / vote_max
    clean_id = torch.where(right_score >= args.theta_s)[0]
    noisy_id = torch.where(right_score < args.theta_s)[0]
    
    clean_id_extended = torch.tensor([], dtype=torch.int64).cuda()
    if len(relabeled_human_labels_score_window.items()) >= args.window_size:
        relabeled_human_labels_score_confidence = torch.mean(torch.stack(relabeled_human_labels_score_window.items()), axis=0)
        logger.log({
            'clean_id_confidence_score_max': torch.max(relabeled_human_labels_score_confidence[clean_id]).detach().item(),
            'clean_id_confidence_score_mean':torch.mean(relabeled_human_labels_score_confidence[clean_id]).detach().item(),
            'clean_id_confidence_score_min': torch.min(relabeled_human_labels_score_confidence[clean_id]).detach().item(),
            'noisy_id_confidence_score_max': torch.max(relabeled_human_labels_score_confidence[noisy_id]).detach().item(),
            'noisy_id_confidence_score_mean':torch.mean(relabeled_human_labels_score_confidence[noisy_id]).detach().item(),
            'noisy_id_confidence_score_min': torch.min(relabeled_human_labels_score_confidence[noisy_id]).detach().item(),
        })
        max_confidence = torch.max(relabeled_human_labels_score_confidence).detach().cpu().item()
        clean_id_extended = torch.where(relabeled_human_labels_score_confidence >= (args.theta_ce * max_confidence))[0]

    return clean_id, clean_id_extended, noisy_id


def select_sample_based_on_sample_stable(feature_bank, labels, args, knn_k, sample_pred_label_window, logger):
    prediction_knn = weighted_knn(feature_bank, feature_bank, labels, args.num_classes, knn_k, 10)  # temperature in weighted KNN
    vote_y = torch.gather(prediction_knn, 1, labels.view(-1, 1)).squeeze()
    vote_max = prediction_knn.max(dim=1)[0]
    right_score = vote_y / vote_max
    clean_id = torch.where(right_score >= args.theta_s)[0]
    noisy_id = torch.where(right_score < args.theta_s)[0]
    
    clean_id_extended = torch.tensor([], dtype=torch.int64).cuda()
    if len(sample_pred_label_window.items()) >= args.window_size:
        stable_labels, ratios = calculate_stabelity_per_sample(sample_pred_label_window)

        logger.log({
            'clean_id_stable_prediction_score_max': torch.max(ratios[clean_id]).detach().item(),
            'clean_id_stable_prediction_score_mean':torch.mean(ratios[clean_id]).detach().item(),
            'clean_id_stable_prediction_score_min': torch.min(ratios[clean_id]).detach().item(),
            'noisy_id_stable_prediction_score_max': torch.max(ratios[noisy_id]).detach().item(),
            'noisy_id_stable_prediction_score_mean':torch.mean(ratios[noisy_id]).detach().item(),
            'noisy_id_stable_prediction_score_min': torch.min(ratios[noisy_id]).detach().item(),
        })

        clean_id_extended = torch.where(ratios >= (args.theta_ce * torch.max(ratios).detach().item()))[0]

    return clean_id, clean_id_extended, noisy_id