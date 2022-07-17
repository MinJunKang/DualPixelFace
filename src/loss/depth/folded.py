
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.geometry import regress_affine, inverse_depth, depth2disp


'''

    Reimplemented Folded Loss of "ICCV 19 paper : Learning Single Camera Depth Estimation Using Dual-Pixels"

'''


class FOLDEDLoss(nn.modules.Module):
    def __init__(self, option):
        super(FOLDEDLoss, self).__init__()
        self.conversion_method = option.dataset.dp_conversion  # given or least_square
        self.weights = torch.tensor(option.model.loss_weight)
        self.num_neighbor_view = option.model.num_neighbor_view
        self.weight_ssim = option.model.weight_ssim
        self.alpha = option.model.alpha
        self.scale = option.model.scale
        self.basic_grid = None
        
    def log1p_safe(self, x):
        """The same as torch.log1p(x), but clamps the input to prevent NaNs."""
        x = torch.as_tensor(x)
        return torch.log1p(torch.min(x, torch.tensor(33e37).to(x)))

    def expm1_safe(self, x):
        """The same as tf.math.expm1(x), but clamps the input to prevent NaNs."""
        x = torch.as_tensor(x)
        return torch.expm1(torch.min(x, torch.tensor(87.5).to(x)))

    def SSIM(self, x, y, conf=None):
        """ Compute the structural similarity index between two images

        Args:
            x: (n_batch, n_dim, nx, ny) input image
            y: (n_batch, n_dim, nx, ny) input image
            conf: (n_batch, n_dim, nx, ny) input confidence map

        Returns:
            (float) structural similarity measure
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d
        
        if conf is not None:
            return torch.clamp((1 - SSIM) / 2, 0, 1) * nn.AvgPool2d(3, 1)(conf)
        else:
            return torch.clamp((1 - SSIM) / 2, 0, 1)

    def CharbonnierLoss(self, x, alpha, scale, conf=None, approximate=False, epsilon=1e-6):
        r"""Implements the general form of the loss.
        This implements the rho(x, \alpha, c) function described in "A General and
        Adaptive Robust Loss Function", Jonathan T. Barron,
        https://arxiv.org/abs/1701.03077.
        Args:
          x: The residual for which the loss is being computed. x can have any shape,
            and alpha and scale will be broadcasted to match x's shape if necessary.
            Must be a tensor of floats.
          conf: confidence map if is possible
          alpha: The shape parameter of the loss (\alpha in the paper), where more
            negative values produce a loss with more robust behavior (outliers "cost"
            less), and more positive values produce a loss with less robust behavior
            (outliers are penalized more heavily). Alpha can be any value in
            [-infinity, infinity], but the gradient of the loss with respect to alpha
            is 0 at -infinity, infinity, 0, and 2. Must be a tensor of floats with the
            same precision as `x`. Varying alpha allows
            for smooth interpolation between a number of discrete robust losses:
            alpha=-Infinity: Welsch/Leclerc Loss.
            alpha=-2: Geman-McClure loss.
            alpha=0: Cauchy/Lortentzian loss.
            alpha=1: Charbonnier/pseudo-Huber loss.
            alpha=2: L2 loss.
          scale: The scale parameter of the loss. When |x| < scale, the loss is an
            L2-like quadratic bowl, and when |x| > scale the loss function takes on a
            different shape according to alpha. Must be a tensor of single-precision
            floats.
          approximate: a bool, where if True, this function returns an approximate and
            faster form of the loss, as described in the appendix of the paper. This
            approximation holds well everywhere except as x and alpha approach zero.
          epsilon: A float that determines how inaccurate the "approximate" version of
            the loss will be. Larger values are less accurate but more numerically
            stable. Must be great than single-precision machine epsilon.
        Returns:
          The losses for each element of x, in the same shape and precision as x.
        """
        assert torch.is_tensor(x)
        assert torch.is_tensor(scale)
        assert torch.is_tensor(alpha)
        assert alpha.dtype == x.dtype
        assert scale.dtype == x.dtype
        assert (scale > 0).all()
        if approximate:
            # `epsilon` must be greater than single-precision machine epsilon.
            assert epsilon > np.finfo(np.float32).eps
            # Compute an approximate form of the loss which is faster, but innacurate
            # when x and alpha are near zero.
            b = torch.abs(alpha - 2) + epsilon
            d = torch.where(alpha >= 0, alpha + epsilon, alpha - epsilon)
            loss = (b / d) * (torch.pow((x / scale) ** 2 / b + 1., 0.5 * d) - 1.)
        else:
            # Compute the exact loss.

            # This will be used repeatedly.
            squared_scaled_x = (x / scale) ** 2

            # The loss when alpha == 2.
            loss_two = 0.5 * squared_scaled_x
            # The loss when alpha == 0.
            loss_zero = self.log1p_safe(0.5 * squared_scaled_x)
            # The loss when alpha == -infinity.
            loss_neginf = -torch.expm1(-0.5 * squared_scaled_x)
            # The loss when alpha == +infinity.
            loss_posinf = self.expm1_safe(0.5 * squared_scaled_x)

            # The loss when not in one of the above special cases.
            machine_epsilon = torch.tensor(np.finfo(np.float32).eps).to(x)
            # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
            beta_safe = torch.max(machine_epsilon, torch.abs(alpha - 2.))
            # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
            alpha_safe = torch.where(alpha >= 0, torch.ones_like(alpha),
                                     -torch.ones_like(alpha)) * torch.max(
                machine_epsilon, torch.abs(alpha))
            loss_otherwise = (beta_safe / alpha_safe) * (
                    torch.pow(squared_scaled_x / beta_safe + 1., 0.5 * alpha) - 1.)

            # Select which of the cases of the loss to return.
            loss = torch.where(
                alpha == -float('inf'), loss_neginf,
                torch.where(
                    alpha == 0, loss_zero,
                    torch.where(
                        alpha == 2, loss_two,
                        torch.where(alpha == float('inf'), loss_posinf,
                                    loss_otherwise))))
        if conf is not None:
            return loss * conf
        else:
            return loss
        
    def grid_maker(self, b, h, w):
        '''
        :param b: batch size
        :param h: fixed cropped image height
        :param w: fixed cropped image width
        :return: cropped image's 2d coordinate grid [B, 3, H, W]
        '''

        if self.basic_grid is None:
            x = np.linspace(0, w - 1, num=int(w))
            y = np.linspace(0, h - 1, num=int(h))
            x_grid, y_grid = np.meshgrid(x, y)
            ones_matrix = np.ones_like(x_grid)
            warp_grid = np.float32(np.stack((x_grid, y_grid, ones_matrix), axis=0))

            self.basic_grid = torch.from_numpy(warp_grid).cuda().unsqueeze(0)  # [1, 3, H, W]

        grid = self.basic_grid.repeat(b, 1, 1, 1)  # [b, 3, H, W]

        return grid
    
    def cropper(self, feature, ch, cw, coord):
        '''
        :param feature: image / feature to be cropped
        :param ch: cropped height
        :param cw: cropped width
        :param coord: [B, 2] : cropped starting coordinate
        :return: batch wise cropped image / feature
        '''

        b, c, h, w = feature.shape
        y_range = torch.arange(0.0, h).repeat(b, 1).cuda() + coord[:, 1].view(b, 1)
        x_range = torch.arange(0.0, w).repeat(b, 1).cuda() + coord[:, 0].view(b, 1)
        yv = y_range.unsqueeze(-1).repeat(1, 1, w)
        xv = x_range.unsqueeze(1).repeat(1, h, 1)

        # normalize
        xv = xv / (w - 1) * 2.0 - 1.0
        yv = yv / (h - 1) * 2.0 - 1.0
        xv = xv.unsqueeze(0)
        yv = yv.unsqueeze(0)

        flow = torch.cat([xv, yv], dim=0).permute([1, 2, 3, 0])

        return F.grid_sample(feature, grid=flow, mode='bilinear', align_corners=False)[:, :, :ch, :cw]
    
    def pixel2cam(self, grid, K, depth, mask=None):
        '''
        :param grid: [B, 3, H, W]
        :param K: [B, 3, 3]
        :param depth: [B, H, W]
        :param mask: [B, H, W]
        :return: [B, 3, H, W]
        '''

        b, h, w = depth.shape

        intrinsic_inv = torch.inverse(K)  # [B, 3, 3]

        cam_coords = intrinsic_inv.bmm(grid.view(b, 3, -1)).view(b, 3, h, w)  # [B, 3, H, W]

        depth = depth.unsqueeze(1)  # [B, 1, H, W]
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, H, W]
            depth[~mask] = 0.0

        return cam_coords * depth

    def cam2pixel(self, tar3dpts, tarP, refP, refK, ref_h, ref_w):
        '''
        :param tar3dpts: [B, 3, H, W]
        :param tarP: [B, 4, 4]
        :param refP: [B, 4, 4]
        :param refK: [B, 3, 3]
        :return: [B, H, W, 2]
        '''

        b, _, h, w = tar3dpts.shape

        # target to reference pose
        tar2refP = refP.bmm(torch.inverse(tarP))  # [B, 4, 4]
        tar2refProj = refK.bmm(tar2refP[:, :3])  # [B, 3, 4]

        # rotation and translation matrix
        rotmat = tar2refProj[:, :, :3]  # [B, 3, 3]
        transmat = tar2refProj[:, :, -1:]  # [B, 3, 1]

        # reference coordinate
        refcoords = rotmat.bmm(tar3dpts.view(b, 3, -1)) + transmat

        # outlier removal
        X = refcoords[:, 0]
        Y = refcoords[:, 1]
        Z = refcoords[:, 2].clamp(min=1e-3)

        # Normalized, -1 if on extreme left, 1 if on extreme right (x = W-1) [B, H*W]
        X_norm = 2 * (X / Z) / (ref_w - 1) - 1
        Y_norm = 2 * (Y / Z) / (ref_h - 1) - 1  # Idem [B, H*W]

        # make sure that no point in warped image is a combinaton of im and gray
        X_mask = (X_norm > 1) + (X_norm < -1) + torch.isinf(X_norm) + torch.isnan(X_norm)
        Y_mask = (Y_norm > 1) + (Y_norm < -1) + torch.isinf(Y_norm) + torch.isnan(Y_norm)
        X_norm[X_mask.detach()] = 2
        Y_norm[Y_mask.detach()] = 2

        pixel_coords = torch.stack([X_norm, Y_norm], dim=-1)

        return pixel_coords.view(b, h, w, 2)

    def warping_operation(self, tarimg, refimgs, tardepth, tarK, tarP, refKs, refPs, coords, selected_idx, tarmask=None):
        '''
        :param selected_idx: randomly selected reference view index
        :param tarimg: [B, 3, H, W] : target raw image
        :param refimgs: [B, N * 3, H, W] : reference raw images
        :param tardepth: [B, CH, CW] : target predicted depth (cropped)
        :param tarK: [B, 3, 3] : target intrinsic matrix K
        :param tarP: [B, 4, 4] : target pose matrix P (world to cam)
        :param refKs: [B, N, 3, 3] : reference intrinsic matrix K
        :param refPs: [B, N, 4, 4] : reference pose matrix P (world to cam)
        :param coords: [B, 2] : cropped starting coordinate
        :param tarmask: [B, CH, CW] : target mask (if None, ignored)
        :return: cropped target raw image and warped cropped reference -> target images
        '''

        wrefimgs = []
        b, _, ref_h, ref_w = refimgs.shape
        refimgs = refimgs.view(b, -1, 3, ref_h, ref_w)
        b, h, w = tardepth.shape

        # get tardepth 3d grid
        grid = self.grid_maker(b, h, w)  # [B, 3, H, W]

        # crop the raw target image
        tarimg = self.cropper(tarimg, h, w, coords)

        # target coordinate to cam coordinate's 3d points
        tar3dpts = self.pixel2cam(grid, tarK, tardepth, tarmask)

        # warp operation
        for i in selected_idx:
            # target 3d points to reference coordinate
            refcoord = self.cam2pixel(tar3dpts, tarP, refPs[:, i], refKs[:, i], ref_h, ref_w)

            # inverse warping
            warped_img = F.grid_sample(refimgs[:, i], grid=refcoord, mode='bilinear', padding_mode='zeros')
            warped_img = warped_img[:, :, :h, :w]

            # masking if possible
            wrefimgs.append(warped_img)

        return tarimg, wrefimgs
    
    def select_random_ref(self, num_img):
        select_num = min(num_img, self.num_neighbor_view)
        arr = np.arange(num_img)
        np.random.shuffle(arr)
        return arr[:select_num].tolist()

    def forward(self, preds, batch, target_type='disp'):

        pred = preds['pred_depth']
        num_pred = pred.shape[1]
        weights = torch.tensor([1.0]).type_as(pred) if num_pred == 1 else self.weights.type_as(pred)
        assert (num_pred == len(weights))
        assert (target_type in ['disp', 'depth', 'idepth'])
        
        mask = batch['mask'] > 0 if 'mask' in batch.keys() else None
        pred_ = pred if target_type in ['disp', 'idepth'] else inverse_depth(pred)
        if self.conversion_method == 'least_square' or not 'abvalue' in batch.keys():
            ab_value = regress_affine(pred[:, 0:1], batch['idepth'].unsqueeze(1))
            gt = depth2disp(batch['depth'].unsqueeze(1), ab_value)
        else:
            ab_value = batch['abvalue']
            gt = batch['disp'] if target_type == 'disp' else batch['idepth']

        conf = None
        if 'conf' in batch.keys():
            if batch['conf'] is not None:
                conf = batch['conf'].unsqueeze(1)

        # type conversion (double to float)
        coords = batch['coords'].float()
        tar_center = batch['raw_center'].float()
        ref_centers = batch['centers'].float()

        # random reference view selection
        selected_idx = self.select_random_ref(refPs.shape[1])

        # warping operation, to test, use target instead of pred[:, 0]
        ctarimg, cwrefimgs = self.warping_operation(tar_center, ref_centers, pred[:, 0],
                                                    batch['K'], batch['P'], batch['Ks'], batch['Ps'], 
                                                    coords, selected_idx, mask)
        loss1 = []
        loss2 = []
        alpha = torch.tensor(self.alpha).float().cuda()
        scale = torch.tensor(self.scale).float().cuda()
        for cwrefimg in cwrefimgs:
            if mask is not None:
                roimask = mask.unsqueeze(1).repeat(1, 3, 1, 1).type_as(ctarimg)
                ctarimg = ctarimg * roimask
                cwrefimg = cwrefimg * roimask
            loss1.append(torch.mean(self.SSIM(ctarimg, cwrefimg, conf)))
            loss2.append(torch.mean(self.CharbonnierLoss(ctarimg - cwrefimg, alpha=alpha, scale=scale, conf=conf)))
        loss1 = sum(loss1) / len(loss1)
        loss2 = sum(loss2) / len(loss2)

        loss = self.weight_ssim * loss1 + (1 - self.weight_ssim) * loss2

        results = {'loss': loss, 'abvalue': abvalue}

        return results