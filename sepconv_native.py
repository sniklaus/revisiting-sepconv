#!/usr/bin/env python

import torch


##########################################################


@torch.compile
def _sepconv_forward(tenIn, tenVer, tenHor):
    intKv, intKh, intH, intW = tenVer.shape[1], tenHor.shape[1], tenVer.shape[2], tenVer.shape[3]

    tenPatches = tenIn[:, :, :intH + intKv - 1, :intW + intKh - 1].unfold(2, intKv, 1).unfold(3, intKh, 1).permute(0, 1, 4, 5, 2, 3)

    if tenIn.shape[1] <= 8:
        tenWeight = tenVer[:, :, None, :, :] * tenHor[:, None, :, :, :]

        return torch.stack([(tenPatches[:, intC] * tenWeight).sum(dim=(1, 2)) for intC in range(tenIn.shape[1])], 1)
    # end

    return (tenVer[:, None, :, None, :, :] * tenHor[:, None, None, :, :, :] * tenPatches).sum(dim=(2, 3))
# end


@torch.compile
def _sepconv_ingrad(tenVer, tenHor, tenOutgrad):
    intKv, intKh, intH, intW = tenVer.shape[1], tenHor.shape[1], tenVer.shape[2], tenVer.shape[3]

    torch._check(tenHor.shape[2] == intH)
    torch._check(tenHor.shape[3] == intW)
    torch._check(tenOutgrad.shape[2] == intH)
    torch._check(tenOutgrad.shape[3] == intW)

    intPad = [intKh - 1, intKh - 1, intKv - 1, intKv - 1]

    tenVerpad = torch.nn.functional.pad(tenVer.flip(1), intPad).unfold(2, intKv, 1).diagonal(dim1=1, dim2=4).unfold(2, intKh, 1).permute(0, 3, 4, 1, 2)
    tenHorpad = torch.nn.functional.pad(tenHor.flip(1), intPad).unfold(3, intKh, 1).diagonal(dim1=1, dim2=4).unfold(1, intKv, 1).permute(0, 4, 3, 1, 2)
    tenOutgradpad = torch.nn.functional.pad(tenOutgrad, intPad).unfold(2, intKv, 1).unfold(3, intKh, 1).permute(0, 1, 4, 5, 2, 3)

    if tenOutgrad.shape[1] <= 8:
        tenWeight = tenVerpad * tenHorpad

        return torch.stack([(tenOutgradpad[:, intC] * tenWeight).sum(dim=(1, 2)) for intC in range(tenOutgrad.shape[1])], 1)
    # end

    return (tenVerpad[:, None] * tenHorpad[:, None] * tenOutgradpad).sum(dim=(2, 3))
# end


@torch.compile
def _sepconv_vergrad(tenIn, tenHor, tenOutgrad, intKv):
    intKh, intH, intW = tenHor.shape[1], tenHor.shape[2], tenHor.shape[3]

    tenPatches = tenIn[:, :, :intH + intKv - 1, :intW + intKh - 1].unfold(2, intKv, 1).unfold(3, intKh, 1)

    if tenIn.shape[1] <= 8:
        tenDot = tenOutgrad[:, 0, :, :, None, None] * tenPatches[:, 0]

        for intC in range(1, tenIn.shape[1]):
            tenDot = tenDot + (tenOutgrad[:, intC, :, :, None, None] * tenPatches[:, intC])
        # end

        return (tenDot * tenHor.permute(0, 2, 3, 1)[:, :, :, None, :]).sum(-1).permute(0, 3, 1, 2).contiguous()
    # end

    return (tenHor[:, None, None, :, :, :] * tenOutgrad[:, :, None, None, :, :] * tenPatches.permute(0, 1, 4, 5, 2, 3)).sum(dim=(1, 3))
# end


@torch.compile
def _sepconv_horgrad(tenIn, tenVer, tenOutgrad, intKh):
    intKv, intH, intW = tenVer.shape[1], tenVer.shape[2], tenVer.shape[3]

    tenPatches = tenIn[:, :, :intH + intKv - 1, :intW + intKh - 1].unfold(2, intKv, 1).unfold(3, intKh, 1)

    if tenIn.shape[1] <= 8:
        tenDot = tenOutgrad[:, 0, :, :, None, None] * tenPatches[:, 0]

        for intC in range(1, tenIn.shape[1]):
            tenDot = tenDot + (tenOutgrad[:, intC, :, :, None, None] * tenPatches[:, intC])
        # end

        return (tenDot.permute(0, 1, 2, 4, 3) * tenVer.permute(0, 2, 3, 1)[:, :, :, None, :]).sum(-1).permute(0, 3, 1, 2).contiguous()
    # end

    return (tenVer[:, None, :, None, :, :] * tenOutgrad[:, :, None, None, :, :] * tenPatches.permute(0, 1, 4, 5, 2, 3)).sum(dim=(1, 2))
# end


##########################################################


class sepconv_func(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, tenIn, tenVer, tenHor):
        tenIn = tenIn.detach().float() # manual cast in case amp is not enabled
        tenVer = tenVer.detach().float() # manual cast in case amp is not enabled
        tenHor = tenHor.detach().float() # manual cast in case amp is not enabled

        tenOut = _sepconv_forward(tenIn, tenVer, tenHor)

        ctx.save_for_backward(tenIn, tenVer, tenHor)

        return tenOut
    # end

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, tenOutgrad):
        tenIn, tenVer, tenHor = ctx.saved_tensors

        tenOutgrad = tenOutgrad.detach().contiguous().float()

        tenIngrad = None
        tenVergrad = None
        tenHorgrad = None

        if ctx.needs_input_grad[0]:
            tenIngrad = _sepconv_ingrad(tenVer, tenHor, tenOutgrad)

            intPadright = tenIn.shape[3] - tenIngrad.shape[3]
            intPadbottom = tenIn.shape[2] - tenIngrad.shape[2]

            if intPadright > 0 or intPadbottom > 0:
                tenIngrad = torch.nn.functional.pad(tenIngrad, [0, intPadright, 0, intPadbottom])
            # end
        # end

        if ctx.needs_input_grad[1]:
            tenVergrad = _sepconv_vergrad(tenIn, tenHor, tenOutgrad, tenVer.shape[1])
        # end

        if ctx.needs_input_grad[2]:
            tenHorgrad = _sepconv_horgrad(tenIn, tenVer, tenOutgrad, tenHor.shape[1])
        # end

        return tenIngrad, tenVergrad, tenHorgrad
    # end
# end
