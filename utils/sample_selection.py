import torch
from .funcs import ExtendedSampleingStrategy
from utils.funcs import * 
# from utils.funcs import get_current_knn_k, FixedSizeQueue
from utils.relabeling import relabel_sample


def select_extended_samples(feature_bank, human_labels, modified_labels, args, knn_k, human_labels_score_window, relabeled_human_labels_score_window):
    if args.extended_sampleing_strategy == ExtendedSampleingStrategy.RELABELD_CONFIDENCE:
        return select_sample_based_on_modified_labales_confidence(feature_bank, modified_labels, args, knn_k, relabeled_human_labels_score_window)

    raise Exception("sampling strategy is not exist")

def select_sample_based_on_modified_labales_confidence(feature_bank, labels, args, knn_k, relabeled_human_labels_score_window):
    prediction_knn = weighted_knn(feature_bank, feature_bank, labels, args.num_classes, knn_k, 10)  # temperature in weighted KNN
    vote_y = torch.gather(prediction_knn, 1, labels.view(-1, 1)).squeeze()
    vote_max = prediction_knn.max(dim=1)[0]
    right_score = vote_y / vote_max
    clean_id = torch.where(right_score >= args.theta_s)[0]
    noisy_id = torch.where(right_score < args.theta_s)[0]
    
    clean_id_extended = torch.tensor([], dtype=torch.int64).cuda()
    if len(relabeled_human_labels_score_window.items()) > args.window_size:
        relabeled_human_labels_score_confidence = torch.mean(torch.stack(relabeled_human_labels_score_window.items()), axis=0)
        clean_id_extended = torch.where(relabeled_human_labels_score_confidence >= args.theta_ce)[0]

    return clean_id, clean_id_extended, noisy_id