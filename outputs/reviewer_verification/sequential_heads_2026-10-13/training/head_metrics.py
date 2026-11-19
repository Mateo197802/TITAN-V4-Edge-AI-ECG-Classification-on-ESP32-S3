"""Freeze the original feature extractor while continuing its two existing heads."""


def head_prefixes(rhythm):
    return ('head_pathology.',)+(('head_rhythm.',) if rhythm else ())


def configure_heads(model,*,rhythm):
    model.eval()
    for name,p in model.named_parameters():p.requires_grad_(name.startswith(head_prefixes(rhythm)))
    model.head_pathology.train()
    if rhythm:model.head_rhythm.train()
    return [p for p in model.parameters() if p.requires_grad]


def assert_heads_frozen(model,initial,*,rhythm):
    import torch

    allowed={name for name,_ in model.named_parameters() if name.startswith(head_prefixes(rhythm))}
    state=model.state_dict()
    if set(state)!=set(initial) or any(state[k].shape!=initial[k].shape for k in state):raise ValueError('Architecture changed')
    changed=[k for k in state if not torch.equal(state[k].detach().cpu(),initial[k])]
    if any(k not in allowed for k in changed):raise ValueError('Frozen parameter or buffer changed')
    return changed
