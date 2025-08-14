import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, \
    precision_recall_curve, auc, r2_score, mean_squared_error
from tqdm import tqdm
# CLAUDE-ENHANCEMENT: Unified AMP import for consistency
from torch import amp

# CLAUDE-COMMENTED-OUT: Old AMP import with try/except
# try:
#     from torch.amp import autocast  # PyTorch >= 1.10
# except ImportError:
#     from torch.cuda.amp import autocast  # PyTorch < 1.10


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader
        self.use_amp = getattr(params, 'use_amp', True)

    def get_metrics_for_multiclass(self, model):
        model.eval()

        truths = []
        preds = []
        for batch in tqdm(self.data_loader, mininterval=1):
            # Support both (x, y) and (x, y, sid)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch

            # (Optional) only skip truly empty batches
            if x.size(0) == 0:
                print("⚠️ Warning: Skipping empty batch with size 0")
                continue

                
            x = x.cuda()
            y = y.cuda()

            # CLAUDE-ENHANCEMENT: Updated AMP usage for consistency
            with amp.autocast('cuda', enabled=self.use_amp):
                pred = model(x)
                
            # CLAUDE-COMMENTED-OUT: Old AMP usage
            # if self.use_amp:
            #     with autocast('cuda'):
            #         pred = model(x)
            # else:
            #     pred = model(x)
            pred_y = torch.max(pred, dim=-1)[1]

            y_np = y.cpu().squeeze().numpy()
            pred_y_np = pred_y.cpu().squeeze().numpy()
            
            # Ensure we have arrays, not scalars
            if y_np.ndim == 0:
                y_np = np.array([y_np])
            if pred_y_np.ndim == 0:
                pred_y_np = np.array([pred_y_np])
                
            truths += y_np.tolist()
            preds += pred_y_np.tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        acc = balanced_accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average='weighted')
        kappa = cohen_kappa_score(truths, preds)
        cm = confusion_matrix(truths, preds)
        return acc, kappa, f1, cm, truths, preds

    def get_metrics_for_binaryclass(self, model):
        model.eval()

        truths = []
        preds = []
        scores = []
        for batch in tqdm(self.data_loader, mininterval=1):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            score_y = torch.sigmoid(pred)
            pred_y = torch.gt(score_y, 0.5).long()
            truths += y.long().cpu().squeeze().numpy().tolist()
            preds += pred_y.cpu().squeeze().numpy().tolist()
            scores += score_y.cpu().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        scores = np.array(scores)
        acc = balanced_accuracy_score(truths, preds)
        roc_auc = roc_auc_score(truths, scores)
        precision, recall, thresholds = precision_recall_curve(truths, scores, pos_label=1)
        pr_auc = auc(recall, precision)
        cm = confusion_matrix(truths, preds)
        return acc, pr_auc, roc_auc, cm

    def get_metrics_for_regression(self, model):
        model.eval()

        truths = []
        preds = []
        for batch in tqdm(self.data_loader, mininterval=1):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            pred = model(x)
            truths += y.cpu().squeeze().numpy().tolist()
            preds += pred.cpu().squeeze().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        corrcoef = np.corrcoef(truths, preds)[0, 1]
        r2 = r2_score(truths, preds)
        rmse = mean_squared_error(truths, preds) ** 0.5
        return corrcoef, r2, rmse