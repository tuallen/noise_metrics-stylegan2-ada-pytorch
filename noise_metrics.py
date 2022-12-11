import torch
import torch_fidelity
import os, json

def add_noise(model, noise_variance=0):
    ''' Adds Gaussian noise to the model parameters.

        Arguments:
            model: PyTorch model
            noise_variance: Variance of the Gaussian noise; adds no noise by default
        Returns: model with Gaussian noise added to the parameters'''

    for param in model.parameters():
        param.add_(torch.randn(param.size()).to(torch.cuda.current_device()) * (noise_variance**2)) # Add Gaussian noise to all parameters

    return model

def calculate_metrics(dir, traindir, noise_variance=None):
    '''Calculate Inception Score (ISC), Frechet Inception Distance (FID), and Kernel Inception Distance (KID)
    for generated images. Saves metrics as desc.json. The variance of the Gaussian noise that was added to
    the model can be added to the file.
    
    Arguments:
        dir: Directory of the generated images
        traindir: Directory of the training images
        noise_variance: Variance of the Gaussian noise added to the weights; not saved by default
    Outputs:
        [dir]/desc.json: Dictionary of ISC, FID, KID, and (optionally) noise_variance
    Returns: True if metrics are saved '''
    
    # Inception-v3 embedding dimensionality is 2048.
    if len(os.listdir(dir)) < 2048 or len(os.listdir(traindir)) < 2048:
        print("The minimum number of images needed is 2048.")
        return False

    # Calculate ISC, FID, and KID
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=dir,
        input2=traindir,
        cuda=True,
        isc=True,
        fid=True,
        kid=True,
        verbose=True,
    )
    if noise_variance is not None: # Add noise_variance to dict if specified
        metrics_dict['noise_variance'] = noise_variance

    # Save metrics as JSON file
    with open(os.path.join(dir,'desc.json'), 'w') as fp:
        json.dump(metrics_dict, fp)
    print("Created " + os.path.join(dir, 'desc.json'))

    return True