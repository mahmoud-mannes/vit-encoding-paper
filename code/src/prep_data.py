def prep_data(corruption_type):
    if corruption_type not in ["JPEG", "Gaussian Blur", None]:
        raise ValueError("corruption_type must be 'JPEG', 'Gaussian Blur', or None")
    # Setting up transforms
    normalize = T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    val_batch_size = 1000
    if not corruption_type:
        transform_val = T.Compose([
            T.Resize(dimensions),
            T.ToTensor(),
            T.ToDtype(torch.float32, scale=True),
            normalize
        ])
        data = Data(dataset_val, transform_val)
    elif corruption_type == "Gaussian Blur":
        GB_transform = T.Compose([
            T.Resize(dimensions),
            T.GaussianBlur(kernel_size=5, sigma=2.5),
            T.ToTensor(),
            T.ToDtype(torch.float32, scale=True),
            normalize
        ])
        data = Data(dataset_val, GB_transform)
    elif corruption_type == "JPEG":
        jpeg_transforms = T.Compose([
            T.Resize(dimensions),
            T.JPEG(quality=5),
            T.ToTensor(),
            T.ToDtype(torch.float32, scale=True),
            normalize
        ])
        data = Data(dataset_val, jpeg_transforms)
    DL = DataLoader(data,val_batch_size,shuffle=False,pin_memory=False, num_workers = min(20,os.cpu_count()))
    return DL