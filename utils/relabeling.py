import torch
from .funcs import RelabelingStrategy

def relabel_sample(
    prediction_cls,
    human_labels,
    args,
    model_prediction_score_window,
    human_labels_score_window
    ):
    if args.relabeling_strategy == RelabelingStrategy.MODEL_CONFIDENCE:
        return relabel_samples_based_on_model_confidence(
            prediction_cls,
            human_labels,
            args,
            model_prediction_score_window,
            human_labels_score_window,
        )

    raise Exception("relabeling strategy is not exist")

def relabel_samples_based_on_model_confidence(
        prediction_cls,
        human_labels,
        args,
        model_prediction_score_window,
        human_labels_score_window,
    ):
    pred_score, pred_label = prediction_cls.max(1)
    print(f'Prediction track: mean: {pred_score.mean()} max: {pred_score.max()} min: {pred_score.min()}')
    
    model_prediction_score_window.enqueue(pred_score)
    human_labels_score = prediction_cls[torch.arange(pred_score.size()[0]), human_labels]
    human_labels_score_window.enqueue(human_labels_score)
    conf_id = torch.tensor([], dtype=torch.int64)
    if (args.relabeling_enable == True) and (len(model_prediction_score_window.items()) >= args.window_size):
        human_labels_score_confidence = torch.mean(torch.stack(human_labels_score_window.items()), dim=0)
        model_prediction_score_confidence = torch.mean(torch.stack(model_prediction_score_window.items()), dim=0)
        max_confidence = torch.max(model_prediction_score_confidence).detach().cpu().item()
        conf_id = torch.where((model_prediction_score_confidence > (args.theta_r * max_confidence)) & (human_labels_score_confidence < (1.0 / args.num_classes)))[0]
        modified_score = torch.clone(human_labels_score).detach()
        modified_label = torch.clone(human_labels).detach()
        modified_score[conf_id] = pred_score[conf_id]
        modified_label[conf_id] = pred_label[conf_id]
        return modified_label, modified_score, conf_id

    return human_labels, human_labels_score, conf_id