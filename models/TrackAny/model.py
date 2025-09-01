

from models.base_model import BaseModel
from .rpn import RPN
from models.TrackAny.RECON import PointTransformer_DAPT


class TrackAny(BaseModel):
    def __init__(self, cfg, log):
        super().__init__(cfg, log)

        self.backbone_net = PointTransformer_DAPT(cfg.backbone_cfg.RECONV)
        self.backbone_net.load_model_from_ckpt('pretrained/base.pth')
        self.loc_net = RPN(cfg.rpn_cfg)

    def forward_embed(self, input):
        pcds = input['pcds']
        mask_refs = input['mask_refs']
        cls_token = input['cls_token']
        bc_refs = input['bc_refs']
        batch_size, duration, npts, _ = pcds.shape

        pcds = pcds.contiguous().view(batch_size * duration, npts, -1)

        b_output = self.backbone_net(pcds,mask_refs,cls_token,bc_refs)
        xyz = b_output['xyz']
        feat = b_output['feat']
        idx = b_output['idx']
        mask_feat = b_output['mask_feat']
        cls_token = b_output['cls_token']
        # mask_preds = b_output['mask_preds']
        assert len(idx.shape) == 2
        return dict(
            xyzs=xyz.view(batch_size, duration, xyz.shape[1], xyz.shape[2]),
            feats=feat,
            idxs=idx.view(batch_size, duration, idx.shape[1]),
            mask_feat=mask_feat,
            cls_token = cls_token,
            # mask_preds = mask_preds
        )

    def forward_localize(self, input):
        # geo_feat mask_feat xyz lwh center_gt
        return self.loc_net(input)

    def forward(self, input, mode):
        forward_dict = dict(
            embed=self.forward_embed,
            localize=self.forward_localize,
        )
        assert mode in forward_dict, '%s has not been supported' % mode

        forward_func = forward_dict[mode]
        output = forward_func(input)
        return output
