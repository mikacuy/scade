import torch
import torch.nn as nn

def reduce_mean_masked_instance(loss, mask_gt):
    # loss: BxK
    loss = torch.where(mask_gt, loss, torch.zeros_like(loss))
    reduced_loss = torch.sum(loss, axis=1) # B
    denom = torch.sum(mask_gt.float(), dim=1) # B
    return torch.where(denom > 0, reduced_loss / denom, torch.zeros_like(reduced_loss)) # B

class MEADSTD_TANH_NORM_Loss(nn.Module):
    """
    loss = MAE((d-u)/s - d') + MAE(tanh(0.01*(d-u)/s) - tanh(0.01*d'))
    """
    def __init__(self, valid_threshold=-1e-8, max_threshold=1e8):
        super(MEADSTD_TANH_NORM_Loss, self).__init__()
        self.valid_threshold = valid_threshold
        self.max_threshold = max_threshold
        #self.thres1 = 0.9

    def transform(self, gt):
        # Get mean and standard deviation
        data_mean = []
        data_std_dev = []
        for i in range(gt.shape[0]):
            gt_i = gt[i]
            mask = gt_i > 0
            depth_valid = gt_i[mask]
            depth_valid = depth_valid[:5]
            if depth_valid.shape[0] < 10:
                data_mean.append(torch.tensor(0).cuda())
                data_std_dev.append(torch.tensor(1).cuda())
                continue
            size = depth_valid.shape[0]
            depth_valid_sort, _ = torch.sort(depth_valid, 0)
            depth_valid_mask = depth_valid_sort[int(size*0.1): -int(size*0.1)]
            data_mean.append(depth_valid_mask.mean())
            data_std_dev.append(depth_valid_mask.std())
        data_mean = torch.stack(data_mean, dim=0).cuda()
        data_std_dev = torch.stack(data_std_dev, dim=0).cuda()

        return data_mean, data_std_dev

    def forward(self, pred, gt):
        """
        Calculate loss.
        """
        mask = (gt > self.valid_threshold) & (gt < self.max_threshold)   # [b, c, h, w]
        mask_sum = torch.sum(mask, dim=(1, 2, 3))
        # mask invalid batches
        mask_batch = mask_sum > 100
        if True not in mask_batch:
            return torch.tensor(0.0, dtype=torch.float).cuda()
        mask_maskbatch = mask[mask_batch]
        pred_maskbatch = pred[mask_batch]
        gt_maskbatch = gt[mask_batch]

        gt_mean, gt_std = self.transform(gt)
        gt_trans = (gt_maskbatch - gt_mean[:, None, None, None]) / (gt_std[:, None, None, None] + 1e-8)

        B, C, H, W = gt_maskbatch.shape
        loss = torch.tensor(0.0).unsqueeze(0).repeat(B).to(pred.device)
        loss_tanh = torch.tensor(0.0).unsqueeze(0).repeat(B).to(pred.device)
        for i in range(B):
            mask_i = mask_maskbatch[i, ...]
            pred_depth_i = pred_maskbatch[i, ...][mask_i]
            gt_trans_i = gt_trans[i, ...][mask_i]

            depth_diff = torch.abs(gt_trans_i - pred_depth_i)

            loss[i] = torch.mean(depth_diff)

            tanh_norm_gt = torch.tanh(0.01*gt_trans_i)
            tanh_norm_pred = torch.tanh(0.01*pred_depth_i)
            loss_tanh[i] = torch.mean(torch.abs(tanh_norm_gt - tanh_norm_pred))

        loss_out = loss + loss_tanh

        return loss_out.float()

        # print(loss_out)

        # ### Reduce mask mean implementation
        # mask_maskbatch = mask_maskbatch.view(B, -1)
        # all_pred = pred_maskbatch.view(B, -1)
        # all_gt = gt_trans.view(B, -1)
        # all_depth_diff = torch.abs(all_gt - all_pred)

        # loss_depth_diff = reduce_mean_masked_instance(all_depth_diff, mask_maskbatch)

        # all_tanh_norm_gt = torch.tanh(0.01*all_gt)
        # all_tanh_norm_pred = torch.tanh(0.01*all_pred)
        # all_tanh_loss = torch.abs(all_tanh_norm_gt - all_tanh_norm_pred)
        # loss_tanh = reduce_mean_masked_instance(all_tanh_loss, mask_maskbatch)

        # loss_out= torch.mean(loss_depth_diff+loss_tanh)

        # if return_per_pixel:
        #     ### For pixel loss map
        #     all_out = all_depth_diff + all_tanh_loss
        #     ## mask out values for invalid pixel depths
        #     all_out = torch.where(mask_maskbatch, all_out, torch.zeros_like(all_out))
        #     all_out = all_out.view(B, C, H, W)
            
        #     return loss_out.float(), all_out
        # else:
        #     return loss_out.float(), None

if __name__ == '__main__':
    ilnr_loss = MEADSTD_TANH_NORM_Loss()
    pred_depth = torch.rand([3, 1, 385, 513]).cuda()
    gt_depth = torch.rand([3, 1, 385, 513]).cuda()

    loss = ilnr_loss(pred_depth, gt_depth)
    print(loss)
