import torch


def convert_tensor_to_display_format(tensor):
    """
    Efficiently converts a PyTorch tensor or a batch of tensors to tensors representing the original images.

    Parameters:
        tensor (torch.Tensor): The input tensor representing an image or a batch of images.
                                Expected shape: (channels, height, width) for single image or
                                                (batch_size, channels, height, width) for batch of images.

    Returns:
        torch.Tensor: A tensor or a batch of tensors representing the original images.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")

    # Check for non-batched input and unsqueeze if necessary to unify handling
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)  # Unsqueeze to make it (1, channels, height, width)
    elif tensor.dim() != 4:
        raise ValueError(
            "Input tensor must have 3 dimensions (channels, height, width) for single images or 4 dimensions (batch_size, channels, height, width) for batched images."
        )

    # Ensure tensor is detached and moved to CPU memory
    tensor = tensor.detach().cpu()

    # Process each image in the (possibly singleton) batch
    processed_images = []
    for img in tensor:
        # Handle grayscale or RGB by transposing accordingly
        if img.shape[0] in (1, 3):
            img = img.squeeze() if img.shape[0] == 1 else img.permute(1, 2, 0)
        else:
            raise ValueError(f"Unsupported channel size: {img.shape[0]}")

        # Convert to uint8 by assuming the input is in [0, 1] and multiplying by 255
        img = img.to(dtype=torch.uint8) * 255
        processed_images.append(img)

    # Reconstruct batched tensor or single tensor based on input
    processed_tensor = torch.stack(processed_images)
    processed_tensor.numpy()
    return processed_tensor if processed_tensor.shape[0] > 1 else processed_tensor.squeeze(0)
