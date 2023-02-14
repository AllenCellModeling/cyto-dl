import torch


def expand_2d_to_3d(seg_prob_tuple, window_data, importance_map_):
    new_3d_tensor = torch.zeros_like(seg_prob_tuple[1]).type_as(seg_prob_tuple[1])
    new_3d_tensor[:, :] = seg_prob_tuple[0].unsqueeze(2)
    seg_prob_tuple = (new_3d_tensor, seg_prob_tuple[1])
    return seg_prob_tuple, importance_map_


def extract_best_z(outs):
    ref_key = "seg"
    key = "mitotic_mask"
    ch = 0
    best_z_seg = torch.argmax(torch.sum(outs[ref_key][ch], dim=(2, 3)), dim=1)
    new_values = []

    for i in range(outs[key].shape[0]):
        new_values.append(outs[key][i, :, best_z_seg[i]])
    outs[key] = torch.stack(new_values)

    return outs
