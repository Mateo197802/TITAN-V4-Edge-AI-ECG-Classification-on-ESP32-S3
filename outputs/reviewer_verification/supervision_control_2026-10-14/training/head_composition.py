"""Recompose independent trained heads only when every shared state tensor matches."""


def compose_head_states(rhythm,pathology):
    import torch

    if set(rhythm)!=set(pathology):raise ValueError('State keys differ')
    if any(rhythm[k].shape!=pathology[k].shape or rhythm[k].dtype!=pathology[k].dtype for k in rhythm):
        raise ValueError('State shape or dtype differs')
    for key in rhythm:
        if not key.startswith(('head_rhythm.','head_pathology.')):
            a=rhythm[key].detach().reshape(-1).contiguous().view(torch.uint8).cpu()
            b=pathology[key].detach().reshape(-1).contiguous().view(torch.uint8).cpu()
            if not torch.equal(a,b):raise ValueError('Shared backbone/buffer/auxiliary state differs: '+key)
    if not any(k.startswith('head_pathology.') for k in pathology):raise ValueError('Missing pathology head')
    return {key:(pathology[key] if key.startswith('head_pathology.') else value).clone()
            for key,value in rhythm.items()}
