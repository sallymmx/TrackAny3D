
import torch.nn.functional as F
import torch
from datasets.utils.pcd_utils import *
from .base_task import BaseTask


class TrackAny(BaseTask):

    def __init__(self, cfg, log):
        super().__init__(cfg, log)

    def build_mask_loss(self, input):
        pred = input['pred']
        gt = input['gt']
        return F.binary_cross_entropy_with_logits(
            pred,
            gt,
        )

    def build_objectness_loss(self, input):
        pred = input['pred']
        gt = input['gt']
        mask = input['mask']
        loss = F.binary_cross_entropy_with_logits(
            pred,
            gt,
            pos_weight=torch.tensor([2.0], device=self.device),
            reduction='none'
        )
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def build_bbox_loss(self, input):
        pred = input['pred']
        gt = input['gt']
        mask = input['mask']
        loss = F.smooth_l1_loss(pred, gt, reduction='none')
        loss = (loss.mean(2) * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def build_center_loss(self, input):
        pred = input['pred']
        gt = input['gt']
        whl = input['whl']
        whl = whl.unsqueeze(1)
        whl = whl.expand(-1, 128, -1)
        mask = input['mask']
        loss = F.mse_loss(pred, gt, reduction='none')
        loss = loss / (whl + 1e-06)

        loss = (loss.mean(2) * mask).sum() / (mask.sum() + 1e-06)
        return loss



    def training_step(self, batch, batch_idx):

        pcds = batch['pcds']  # b,t,n,3\
        search_t1_pcd = pcds[:,1,:,:]
        search_t2_pcd = pcds[:,2,:,:]
        mask_refs = batch['mask_refs']  # b,t,n 50 2 1024
        mask_gts = batch['mask_gts']  # b,t,n

        bbox_gts = batch['bbox_gts']  # b,t,4
        bc_refs = batch['bc_refs']
        first_mask_gt = batch['first_mask_gt']  # b,n
        first_bbox_gt = batch['first_bbox_gt']  # b,4
        is_dynamic_gts = batch['is_dynamic_gts']  # b,t
        lwh = batch['lwh']  # b,3
        whl = batch['whl']
        n_smp_frame = self.cfg.dataset_cfg.num_smp_frames_per_tracklet
        cls_token = None
        for i in range(1, n_smp_frame):

            embed_output = self.model(dict(
                pcds=pcds[:,[0,i],:,:],
                cls_token = cls_token,
                mask_refs=mask_refs[:,[0,i],:],
                bc_refs=bc_refs[:, [0, i], :],
            ), mode='embed')

            xyzs, geo_feat, idxs,mask_feat,cls_token = embed_output['xyzs'], embed_output['feats'], embed_output['idxs'],embed_output['mask_feat'],embed_output['cls_token']






            mask_loss, crs_obj_loss, rfn_obj_loss, center_loss, bbox_loss = 0.0, 0.0, 0.0, 0.0, 0.0


            localize_output = self.model(dict(
                geo_feat=geo_feat,
                mask_feat=mask_feat,
                xyz=xyzs[:, 1, :, :],
                lwh=lwh,
                center_gt=bbox_gts[:, i, :3],
                whl = whl,
            ), mode='localize')
            mask_pred = localize_output['mask_pred']



            # for k in list(mask_preds.keys()):
            #     if k.startswith('search_mask_pred_'):
            #         s = int(k.split('_')[-1])
            #
            #         t_m_pd = mask_preds.pop('search_mask_pred_%d' % s)
            #
            #         loss_cascaded_mask += F.binary_cross_entropy_with_logits(
            #             t_m_pd,
            #             torch.gather(mask_gts[:, i, :], 1, idxs[:, 1, :]),
            #             pos_weight=torch.tensor([1.0], device=self.device)
            #         )


            # b,n
            mask_loss += self.build_mask_loss(dict(
                pred=mask_pred,
                gt=torch.gather(mask_gts[:, i, :], 1, idxs[:, 1, :])
            ))
            center_pred = localize_output['center_pred']
            offset_center_pred = localize_output['offset_center_pred']

            center_loss += self.build_center_loss(dict(
                whl = whl,
                pred=offset_center_pred,
                gt=bbox_gts[:, i, :3].unsqueeze(
                    1).expand_as(center_pred)-xyzs[:,1,:,:],
                mask=torch.gather(mask_gts[:, i, :], 1, idxs[:, 1, :])
            ))

            dist = torch.sum(
                (center_pred - bbox_gts[:, i, None, :3]) ** 2, dim=-1)
            dist = torch.sqrt(dist + 1e-6)  # B, K
            objectness_label = torch.zeros_like(dist, dtype=torch.float)
            objectness_label[dist < 0.3] = 1
            objectness_mask = torch.ones_like(
                objectness_label, dtype=torch.float)
            objectness_pred = localize_output['objectness_pred']
            crs_obj_loss += self.build_objectness_loss(dict(
                pred=objectness_pred,
                gt=objectness_label,
                mask=objectness_mask
            ))

            bboxes_pred = localize_output['bboxes_pred']
            proposal_xyz = localize_output['proposal_xyz']
            dist = torch.sum(
                (proposal_xyz - bbox_gts[:, i, None, :3]) ** 2, dim=-1)

            dist = torch.sqrt(dist + 1e-6)  # B, K
            objectness_label = torch.zeros_like(dist, dtype=torch.float)
            objectness_label[dist < 0.3] = 1
            objectness_pred = bboxes_pred[:, :, 4]  # B, K
            objectness_mask = torch.ones_like(
                objectness_label, dtype=torch.float)
            rfn_obj_loss += self.build_objectness_loss(dict(
                pred=objectness_pred,
                gt=objectness_label,
                mask=objectness_mask
            ))
            bbox_loss += self.build_bbox_loss(dict(
                pred=bboxes_pred[:, :, :4],
                gt=bbox_gts[:, i, None, :4].expand_as(
                    bboxes_pred[:, :, :4]),
                mask=objectness_label
            ))



        loss = self.cfg.loss_cfg.mask_weight * mask_loss + \
            self.cfg.loss_cfg.crs_obj_weight * crs_obj_loss + \
            self.cfg.loss_cfg.rfn_obj_weight * rfn_obj_loss + \
            self.cfg.loss_cfg.bbox_weight * bbox_loss + \
            self.cfg.loss_cfg.center_weight * center_loss
        # loss.backward()
        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         print(name)

        self.logger.experiment.add_scalars(
            'loss',
            {
                'loss_total': loss,
                'loss_bbox': bbox_loss,
                'loss_center': center_loss,
                'loss_mask': mask_loss,
                'loss_rfn_objectness': rfn_obj_loss,
                'loss_crs_objectness': crs_obj_loss,
                # 'loss_cascaded_mask' : loss_cascaded_mask
            },
            global_step=self.global_step
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _to_float_tensor(self, data):
        tensor_data = {}
        for k, v in data.items():
            tensor_data[k] = torch.tensor(
                v, device=self.device, dtype=torch.float32).unsqueeze(0)
        return tensor_data

    def forward_on_tracklet(self, tracklet):

        pred_bboxes = []
        gt_bboxes = []

        memory = None
        lwh = None
        whl = None
        last_bbox_cpu = np.array([0.0, 0.0, 0.0, 0.0])

        for frame_id, frame in enumerate(tracklet):
            prev_id = max(frame_id-1,0)
            gt_bboxes.append(frame['bbox'])
            if frame_id == 0:
                base_bbox = frame['bbox']
                lwh = np.array(
                    [base_bbox.wlh[1], base_bbox.wlh[0], base_bbox.wlh[2]])

                whl = np.array(
                    [base_bbox.wlh[0], base_bbox.wlh[2], base_bbox.wlh[1]])
            else:
                base_bbox = pred_bboxes[-1]

            pcd = crop_and_center_pcd(
                frame['pcd'], base_bbox, offset=self.cfg.dataset_cfg.frame_offset, offset2=self.cfg.dataset_cfg.frame_offset2, scale=self.cfg.dataset_cfg.frame_scale)
            template_pcd,template_bbox_ref = crop_and_center_pcd(
                tracklet[prev_id]['pcd'], base_bbox, offset=self.cfg.dataset_cfg.frame_offset, offset2=self.cfg.dataset_cfg.frame_offset2, scale=self.cfg.dataset_cfg.frame_scale,return_box=True)

            if frame_id == 0:
                # print(pcd.nbr_points())
                if pcd.nbr_points() == 0:
                    pcd.points = np.array([[0.0],[0.0],[0.0]])



                pred_bboxes.append(frame['bbox'])
                continue
                # print(mask_gt.shape, pcd.nbr_points())
            else:
                if pcd.nbr_points() <= 1:
                    bbox = get_offset_box(
                        pred_bboxes[-1], last_bbox_cpu, use_z=self.cfg.dataset_cfg.eval_cfg.use_z, is_training=False)
                    pred_bboxes.append(bbox)
                    continue

                pcd, idx = resample_pcd(
                    pcd, self.cfg.dataset_cfg.frame_npts, return_idx=True, is_training=False)
                search_mask_ref = np.ones([pcd.points.shape[1], ]) * 0.5
                search_bc_ref = np.zeros([pcd.points.shape[1], 9])
                template_pcd, template_idx = resample_pcd(
                    template_pcd, self.cfg.dataset_cfg.frame_npts, return_idx=True, is_training=False)

                template_mask_ref = get_pcd_in_box_mask(
                    template_pcd, template_bbox_ref, scale=1.25).astype('float32')
                template_bc_ref = get_point_to_box_distance(
                    template_pcd, template_bbox_ref)
                if frame_id != 1:
                    template_mask_ref[template_mask_ref == 0] = 0.2
                    template_mask_ref[template_mask_ref == 1] = 0.8

            template_mask_ref = torch.tensor(template_mask_ref).unsqueeze(0).unsqueeze(1)
            search_mask_ref = torch.tensor(search_mask_ref).unsqueeze(0).unsqueeze(1)
            template_bc_ref = torch.tensor(template_bc_ref).unsqueeze(0).unsqueeze(1)
            search_bc_ref = torch.tensor(search_bc_ref).unsqueeze(0).unsqueeze(1)
            mask_refs = torch.cat([template_mask_ref,search_mask_ref],dim=1)
            bc_refs = torch.cat([template_bc_ref,search_bc_ref],dim=1)
            pcd = torch.tensor(pcd.points.T, device=self.device,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            template_pcd = torch.tensor(template_pcd.points.T, device=self.device,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            pcds = torch.cat((template_pcd,pcd), dim=1)
            cls_token = None

            embed_output = self.model(dict(
                pcds=pcds,
                mask_refs=mask_refs.to(self.device),
                bc_refs=bc_refs.float().to(self.device),
                cls_token = cls_token
            ), mode='embed')


            xyzs, geo_feat, idxs,mask_feat,cls_token = embed_output['xyzs'], embed_output['feats'], embed_output['idxs'],embed_output['mask_feat'],embed_output['cls_token']

            if frame_id == 0:

                pred_bboxes.append(frame['bbox'])
            else:


                localize_output = self.model(dict(
                    geo_feat=geo_feat,
                    mask_feat=mask_feat,
                    xyz=xyzs[:, 1, :, :],
                    lwh=torch.tensor(lwh, device=self.device,
                                     dtype=torch.float32).unsqueeze(0),
                    whl=torch.tensor(whl, device=self.device,
                                     dtype=torch.float32).unsqueeze(0),
                ), mode='localize')
                mask_pred = localize_output['mask_pred']
                bboxes_pred = localize_output['bboxes_pred']
                bboxes_pred_cpu = bboxes_pred.squeeze(
                    0).detach().cpu().numpy()

                bboxes_pred_cpu[np.isnan(bboxes_pred_cpu)] = -1e6
                # remove bboxes whose objectness pred is nan
                # it may happen at the early stage of training

                best_box_idx = bboxes_pred_cpu[:, 4].argmax()
                bbox_cpu = bboxes_pred_cpu[best_box_idx, 0:4]
                if torch.max(mask_pred.sigmoid()) < self.cfg.missing_threshold:
                    bbox = get_offset_box(
                        pred_bboxes[-1], last_bbox_cpu, use_z=self.cfg.dataset_cfg.eval_cfg.use_z, is_training=False)
                else:
                    bbox = get_offset_box(
                        pred_bboxes[-1], bbox_cpu, use_z=self.cfg.dataset_cfg.eval_cfg.use_z, is_training=False)
                    last_bbox_cpu = bbox_cpu

                pred_bboxes.append(bbox)

        return pred_bboxes, gt_bboxes
