import argparse

import torch.optim.lr_scheduler
import torchvision.transforms as transforms
import wandb

from torch.optim import SGD
from torch.utils.data import Subset
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt

from datasets.dataloader_animal10n import animal_dataset
from torchvision.models import vgg19_bn
from utils import *
from utils.funcs import get_current_knn_k, FixedSizeQueue
from utils.relabeling import relabel_sample
from utils.sample_selection import  select_extended_samples


parser = argparse.ArgumentParser('Train with ANIMAL-10N dataset')
parser.add_argument('--dataset_path', default='ANIMAL-10N', help='dataset path')

# model settings
parser.add_argument('--lambda_fc',  default=1.0, type=float, help='weight of feature consistency loss (default: 1.0)')
parser.add_argument('--theta_s',    default=1.0, type=float, help='threshold for voted correct samples (default: 1.0)')
parser.add_argument('--theta_r',    default=0.95, type=float, help='threshold for relabel samples (default: 0.95)')
parser.add_argument('--theta_ce',   default=0.95, type=float, help='threshold for clean extended samples (default: 0.95)')
parser.add_argument('--k',          default=200, type=int, help='neighbors for soft-voting (default: 200)')

# train settings
parser.add_argument('--window_size',    default=20, type=int, metavar='N', help='number of total epochs to run (default: 5)')
parser.add_argument('--epochs',         default=150, type=int, metavar='N', help='number of total epochs to run (default: 200)')
parser.add_argument('--batch_size',     default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr',             default=0.02, type=float, help='initial learning rate (default: 0.02)')
parser.add_argument('--momentum',       default=0.9, type=float, metavar='M', help='momentum of SGD solver (default: 0.9)')
parser.add_argument('--weight_decay',   default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--seed',           default=3047, type=int, help='seed for initializing training. (default: 3047)')
parser.add_argument('--gpuid',          default='0', type=str, help='Selected GPU (default: "0")')
parser.add_argument('--entity', type=str, help='Wandb user entity')
parser.add_argument('--run_path', type=str, help='run path containing all results')
parser.add_argument('--extend_clean_knn_samples_enable',     default=True, type=bool,  help='extend_clean_knn_samples_enable')
parser.add_argument('--decrease_knn_k_enable', type=bool,    default=False, help='decrease_knn_k_enable')
parser.add_argument('--knn_k_decrease_rate', type=float,     default=1.0025, help='knn_k_decrease_rate')
parser.add_argument('--min_knn_k', type=int,                 default=200, help='min_knn_k')
parser.add_argument('--ca_warm_restarts_enabled', type=bool, default=False, help='ca_warm_restarts_enabled')
parser.add_argument('--relabeling_strategy', type=str, default="model_confidence", help='relabeling_strategy')
parser.add_argument('--extended_sampleing_strategy', type=str, default="relabeld_confidence", help='relabeling_strategy')

def train(labeled_trainloader, modified_label, all_trainloader, encoder, classifier, proj_head, pred_head, optimizer, epoch, args):
    encoder.train()
    classifier.train()
    proj_head.train()
    pred_head.train()
    xlosses = AverageMeter('xloss')
    ulosses = AverageMeter('uloss')
    labeled_train_iter = iter(labeled_trainloader)
    all_bar = tqdm(all_trainloader)
    for batch_idx, ([inputs_u1, inputs_u2],  _, _) in enumerate(all_bar):
        try:
            [inputs_x1, inputs_x2], labels_x, index = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            [inputs_x1, inputs_x2], labels_x, index = next(labeled_train_iter)

        # cross-entropy training with mixup
        batch_size = inputs_x1.size(0)
        inputs_x1, inputs_x2 = inputs_x1.cuda(), inputs_x2.cuda()
        labels_x = modified_label[index]
        targets_x = torch.zeros(batch_size, args.num_classes, device=inputs_x1.device).scatter_(1, labels_x.view(-1, 1), 1)
        l = np.random.beta(0.5, 0.5)
        l = max(l, 1 - l)
        all_inputs_x = torch.cat([inputs_x1, inputs_x2], dim=0)
        all_targets_x = torch.cat([targets_x, targets_x], dim=0)
        idx = torch.randperm(all_inputs_x.size()[0])
        input_a, input_b = all_inputs_x, all_inputs_x[idx]
        target_a, target_b = all_targets_x, all_targets_x[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits = classifier(encoder(mixed_input))
        Lce = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))

        # optional feature-consistency
        inputs_u1, inputs_u2 = inputs_u1.cuda(), inputs_u2.cuda()

        feats_u1 = encoder(inputs_u1)
        feats_u2 = encoder(inputs_u2)
        f, h = proj_head, pred_head

        z1, z2 = f(feats_u1), f(feats_u2)
        p1, p2 = h(z1), h(z2)
        Lfc = D(p2, z1)
        loss = Lce + args.lambda_fc * Lfc
        xlosses.update(Lce.item())
        ulosses.update(Lfc.item())
        all_bar.set_description(
            f'Train epoch {epoch} LR:{optimizer.param_groups[0]["lr"]} Labeled loss: {xlosses.avg:.4f} Unlabeled loss: {ulosses.avg:.4f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logger.log({'ce loss': xlosses.avg, 'fc loss': ulosses.avg})


def test(testloader, encoder, classifier, epoch):
    encoder.eval()
    classifier.eval()
    accuracy = AverageMeter('accuracy')
    data_bar = tqdm(testloader)
    with torch.no_grad():
        for i, (data, label, _) in enumerate(data_bar):
            data, label = data.cuda(), label.cuda()
            feat = encoder(data)
            res = classifier(feat)
            pred = torch.argmax(res, dim=1)
            acc = torch.sum(pred == label) / float(data.size(0))
            accuracy.update(acc.item(), data.size(0))
            data_bar.set_description(f'Test epoch {epoch}: Accuracy#{accuracy.avg:.4f}')
    logger.log({'acc': accuracy.avg})
    return accuracy.avg


def evaluate(dataloader, encoder, classifier, args, human_labels, knn_k,
    model_prediction_score_window,
    human_labels_score_window,
    relabeled_human_labels_score_window
):
    encoder.eval()
    classifier.eval()
    feature_bank = []
    prediction = []
    # targets = []

    ################################### feature extraction ###################################
    with torch.no_grad():
        # generate feature bank
        for (data, target, index) in tqdm(dataloader, desc='Feature extracting'):
            data = data.cuda()
            feature = encoder(data)
            feature_bank.append(feature)
            res = classifier(feature)
            prediction.append(res)
            # targets.append(target)
        # targets = torch.cat(targets, dim=0)
        feature_bank = F.normalize(torch.cat(feature_bank, dim=0), dim=1)
        prediction_cls = torch.softmax(torch.cat(prediction, dim=0), dim=1)
        ################################### sample relabelling ###################################
        # his_score, pred_label = prediction_cls.max(1)
        modified_label, modified_score, changed_id = relabel_sample(
                                        prediction_cls, human_labels, args,
                                        model_prediction_score_window,
                                        human_labels_score_window,)
        
        relabeled_human_labels_score_window.enqueue(modified_score)
        ################################### sample selection ###################################
        clean_id, clean_id_extended, noisy_id = select_extended_samples(
            feature_bank, human_labels, modified_label, args, knn_k, 
            human_labels_score_window, relabeled_human_labels_score_window )

    return clean_id, clean_id_extended, noisy_id, modified_label, changed_id


def main():
    args = parser.parse_args()
    seed_everything(args.seed)
    if args.run_path is None:
        args.run_path = f'Dataset(animal10n_Model({args.theta_r}_{args.theta_s})'

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    global logger
    model_prediction_score_window = FixedSizeQueue(args.window_size)
    relabeled_human_labels_score_window = FixedSizeQueue(args.window_size)
    human_labels_score_window = FixedSizeQueue(args.window_size)

    logger = wandb.init(project='animal10n', entity=args.entity, name=args.run_path)
    logger.config.update(args)

    if not os.path.isdir(f'animal10n'):
        os.mkdir(f'animal10n')
    if not os.path.isdir(f'animal10n/{args.run_path}'):
        os.mkdir(f'animal10n/{args.run_path}')

    ############################# Dataset initialization ##############################################
    args.num_classes = 10
    args.image_size = 64

    # data loading
    weak_transform = transforms.Compose([
        transforms.RandomCrop(args.image_size, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    none_transform = transforms.Compose([transforms.ToTensor()])  # no augmentation
    strong_transform = transforms.Compose([transforms.RandomCrop(args.image_size, padding=8),
                                           transforms.RandomHorizontalFlip(),
                                           RandAugment(),
                                           transforms.ToTensor()])

    # eval data served as soft-voting pool
    train_data = animal_dataset(root=args.dataset_path, transform=KCropsTransform(strong_transform, 2), mode='train')
    eval_data = animal_dataset(root=args.dataset_path, transform=weak_transform, mode='train')
    test_data = animal_dataset(root=args.dataset_path, transform=none_transform, mode='test')
    all_data = animal_dataset(root=args.dataset_path, transform=MixTransform(strong_transform, weak_transform, 1), mode='train')

    # noisy labels
    human_labels = torch.tensor(eval_data.targets).cuda()

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_loader = torch.utils.data.DataLoader(all_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    ################################ Model initialization ###########################################
    encoder = vgg19_bn(pretrained=False)
    encoder.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(p=0.2),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(p=0.2),
    )

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0, 0.005)
            nn.init.constant_(m.bias, 0)

    encoder.classifier.apply(init_weights)
    classifier = nn.Linear(4096, args.num_classes)
    proj_head = torch.nn.Sequential(torch.nn.Linear(4096, 256),
                                    torch.nn.BatchNorm1d(256),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 128))
    pred_head = torch.nn.Sequential(torch.nn.Linear(128, 256),
                                    torch.nn.BatchNorm1d(256),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 128))
    classifier.apply(init_weights)

    encoder.cuda()
    classifier.cuda()
    proj_head.cuda()
    pred_head.cuda()

    #################################### Training initialization #######################################
    optimizer = SGD([{'params': encoder.parameters()}, {'params': classifier.parameters()}, {'params': proj_head.parameters()}, {'params': pred_head.parameters()}],
                    lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
    warmup_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=args.lr / 5, last_epoch=-1)
    endup_schedulerx = None
    warmup_epochs = 10
    acc_logs = open(f'animal10n/{args.run_path}/acc.txt', 'w')
    save_config(args, f'animal10n/{args.run_path}')
    print('Train args: \n', args)
    best_acc = 0
    knn_k = args.k

    ################################ Training loop ###########################################
    for i in range(args.epochs):
        clean_id, clean_id_extended, noisy_id, modified_label, relabel_ids = evaluate(
            eval_loader, encoder, classifier, args, human_labels, knn_k,
            model_prediction_score_window, human_labels_score_window, relabeled_human_labels_score_window)

        if args.extend_clean_knn_samples_enable == True:
            union_set = torch.unique(torch.cat((clean_id, clean_id_extended)))
            number_of_extended_samples = len(union_set) - len(clean_id)
            logger.log({'number_of_extended_samples': number_of_extended_samples,
                        'number_of_union_set_samples': len(union_set)})
            clean_subset = Subset(train_data, union_set.cpu())
            sampler = ClassBalancedSampler(labels=modified_label[union_set], num_classes=args.num_classes)
        else:
            clean_subset = Subset(train_data, clean_id.cpu())
            sampler = ClassBalancedSampler(labels=modified_label[clean_id], num_classes=args.num_classes)

        labeled_loader = torch.utils.data.DataLoader(clean_subset, batch_size=args.batch_size, sampler=sampler, num_workers=4, drop_last=True)

        train(labeled_loader, modified_label, all_loader, encoder, classifier, proj_head, pred_head, optimizer, i, args)

        cur_acc = test(test_loader, encoder, classifier, i)

        if (i < 10) and (args.ca_warm_restarts_enabled == True):
            warmup_scheduler.step()
        elif (i > 0.9 * args.epochs) and (args.ca_warm_restarts_enabled == True):
            if endup_schedulerx == None:
                optimizer = torch.optim.SGD(
                    [{'params': encoder.parameters()}, {'params': classifier.parameters()}, {'params': proj_head.parameters()}, {'params': pred_head.parameters()}]
                    , lr=optimizer.param_groups[0]["lr"] * 2, weight_decay=args.weight_decay, momentum=args.momentum)
                endup_schedulerx = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=0.00001, last_epoch=-1)
            endup_schedulerx.step()
        else:
            scheduler.step()

        if args.decrease_knn_k_enable == True:
            knn_k = get_current_knn_k(args.k, i, args.knn_k_decrease_rate, args.min_knn_k)

        logger.log({'knn_k': knn_k, 'clean samples': len(clean_subset), 'lr': optimizer.param_groups[0]["lr"],  'relabeled samples': len(relabel_ids)})
        if cur_acc > best_acc:
            best_acc = cur_acc
            save_checkpoint({
                'cur_epoch': i,
                'classifier': classifier.state_dict(),
                'encoder': encoder.state_dict(),
                'proj_head': proj_head.state_dict(),
                'pred_head': pred_head.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=f'animal10n/{args.run_path}/best_acc.pth.tar')
        acc_logs.write(f'Epoch [{i}/{args.epochs}]: Best accuracy@{best_acc}! Current accuracy@{cur_acc} \n')
        acc_logs.flush()
        print(f'Epoch [{i}/{args.epochs}]: Best accuracy@{best_acc}! Current accuracy@{cur_acc} \n')
    save_checkpoint({
        'cur_epoch': i,
        'classifier': classifier.state_dict(),
        'encoder': encoder.state_dict(),
        'proj_head': proj_head.state_dict(),
        'pred_head': pred_head.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename=f'animal10n/{args.run_path}/last.pth.tar')


if __name__ == '__main__':
    main()
