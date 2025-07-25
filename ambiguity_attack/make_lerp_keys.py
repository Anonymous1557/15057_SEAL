import numpy as np
import os
import argparse

def generate_random_noise(shape):
    return np.random.randn(*shape)

def blend_and_save_npy(original_path, output_dir):
    original = np.load(original_path)

    noise = generate_random_noise(original.shape)

    blend_ratios = np.linspace(0, 1, 11)
    
    original_filename = os.path.basename(original_path).split('.')[0]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for ratio in blend_ratios:
        blended = (1 - ratio) * original + ratio * noise
        
        output_filename = f"{original_filename}_blend_{ratio:.1f}.npy"
        output_path = os.path.join(output_dir, output_filename)
        
        np.save(output_path, blended)
        print(f"Saved blended file: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Blend original npy file with random noise and save the results.')
    parser.add_argument('original_path', type=str, help='Path to the original npy file')
    parser.add_argument('output_dir', type=str, help='Directory to save the blended npy files')

    args = parser.parse_args()
    
    blend_and_save_npy(args.original_path, args.output_dir)
