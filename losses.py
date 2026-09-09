"""Training objectives for SINR pretraining and SACF-SINR."""

import torch
import torch.nn.functional as F

import utils


def get_loss_function(params):
    loss_name = params.get("loss", "an_full")
    if loss_name == "an_full":
        return an_full
    if loss_name == "sacf_context":
        return sacf_context
    raise ValueError(f"Unknown loss: {loss_name}")


def neg_log(values):
    return -torch.log(values + 1e-5)


def an_full(batch, model, params, loc_to_feats):
    """Assume-negative loss used to pretrain the SINR location encoder."""
    indices = torch.arange(params["batch_size"])
    location_features, _, class_id = batch
    location_features = location_features.to(params["device"])
    class_id = class_id.to(params["device"])
    batch_size = location_features.shape[0]

    random_locations = utils.sample_bg_locations(batch_size, params)
    random_features = loc_to_feats(random_locations, normalize=False)
    embeddings = model(
        torch.cat((location_features, random_features), 0), return_feats=True
    )
    observed = embeddings[:batch_size]
    background = embeddings[batch_size:]
    observed_probability = torch.sigmoid(model.class_emb(observed))
    background_probability = torch.sigmoid(model.class_emb(background))

    observed_loss = neg_log(1.0 - observed_probability)
    background_loss = neg_log(1.0 - background_probability)

    observed_loss[indices[:batch_size], class_id] = params.get(
        "pos_weight", 2048
    ) * neg_log(observed_probability[indices[:batch_size], class_id])
    return observed_loss.mean() + background_loss.mean()


def _reduce_field_scores(field_scores):
    return torch.logsumexp(field_scores, dim=2)


def _sample_support(model, species, query_locations, params, device):
    batch_size = len(species)
    max_context = int(params.get("ctx_k", 20))
    counts = params["ctx_pool_count"][species]
    draw_width = max(max_context, int(counts.max()))

    columns = torch.arange(draw_width, device=device).unsqueeze(0)
    safe_counts = counts.clamp(min=1)
    all_indices = params["ctx_pool_start"][species].unsqueeze(1) + columns.clamp(
        max=(safe_counts - 1).unsqueeze(1)
    )
    all_locations = params["ctx_pool_locs"][all_indices.reshape(-1)].reshape(
        batch_size, draw_width, -1
    )
    candidates = columns < counts.unsqueeze(1)
    candidates &= ~(all_locations == query_locations.unsqueeze(1)).all(-1)
    random_keys = torch.rand(batch_size, draw_width, device=device)
    random_keys = random_keys.masked_fill(~candidates, float("inf"))
    offsets = random_keys.topk(max_context, largest=False, dim=1).indices
    offsets = offsets.sort(dim=1).values
    selected_valid = candidates.gather(1, offsets)
    selected_indices = params["ctx_pool_start"][species].unsqueeze(1) + offsets.clamp(
        max=(safe_counts - 1).unsqueeze(1)
    )
    selected_features = params["ctx_pool_feats"][selected_indices.reshape(-1)].reshape(
        batch_size, max_context, -1
    )
    tokens = model.encode_location(selected_features)
    valid = (counts > 0).unsqueeze(1).expand(batch_size, max_context).clone()
    valid &= selected_valid
    valid &= torch.rand(batch_size, max_context, device=device) >= params.get(
        "ctx_loc_token_drop", 0.3
    )
    return tokens, valid


def _metadata_availability(model, species, support_present, regime):
    content_tokens, has_text, has_image = model.build_content_tokens(species)
    only_location = regime < 0.1
    only_text = (regime >= 0.1) & (regime < 0.2)
    only_image = (regime >= 0.2) & (regime < 0.3)
    location_text = (regime >= 0.3) & (regime < 0.4)
    location_image = (regime >= 0.4) & (regime < 0.5)
    use_text = ~(only_location | only_image | location_image)
    use_image = ~(only_location | only_text | location_text)
    use_text &= has_text
    use_image &= has_image
    use_text |= use_image.logical_not() & has_text & ~use_text
    use_image |= use_text.logical_not() & has_image & ~use_image
    use_text |= ~(use_text | use_image) & ~support_present
    return content_tokens, torch.stack((use_text, use_image), dim=1)


def _lanb_terms(data_scores, random_scores, class_id, positive_weight):
    data_scores = data_scores.float()
    random_scores = random_scores.float()
    batch_size = data_scores.shape[0]
    indices = torch.arange(batch_size, device=data_scores.device)
    same_species = class_id[:, None] == class_id[None, :]
    matrix = F.softplus(data_scores) * (~same_species).float()
    matrix[indices, indices] = float(positive_weight) * F.softplus(
        -data_scores[indices, indices]
    )
    return matrix, F.softplus(random_scores)


def sacf_context(batch, model, params, loc_to_feats):
    """Train SACF-SINR with modality sampling and the LANB objective."""
    location_features, locations, class_id = batch
    device = params["device"]
    location_features = location_features.to(device)
    locations = locations.to(device)
    class_id = class_id.to(device)
    batch_size = location_features.shape[0]

    random_locations = utils.sample_bg_locations(batch_size, params)
    random_features = loc_to_feats(random_locations, normalize=False).to(device)
    position_tokens = model.encode_location(
        torch.cat((location_features, random_features), dim=0)
    )
    context_tokens, context_valid = _sample_support(
        model, class_id, locations, params, device
    )

    regime = torch.rand(batch_size, device=device)
    no_location = ((regime >= 0.1) & (regime < 0.3)) | (
        (regime >= 0.5) & (regime < 0.6)
    )
    context_valid[no_location] = False
    support_present = context_valid.any(1)
    content_tokens, content_valid = _metadata_availability(
        model, class_id, support_present, regime
    )

    fields = model.species_emb.context_tf(
        context_tokens, context_valid, content_tokens, content_valid
    )
    scores = _reduce_field_scores(torch.einsum("bd,ukd->buk", position_tokens, fields))
    data_scores = scores[:batch_size]
    random_scores = scores[batch_size:]
    matrix, background = _lanb_terms(
        data_scores,
        random_scores,
        class_id,
        params.get("pos_weight", 2048),
    )
    return matrix.mean() + background.mean()
