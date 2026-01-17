
import torch
from torch.utils.data import DataLoader
from model.tempo import build_tempo
from data.data_vimeo_triplet import Vimeo90KTriplet, vimeo_collate
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def run_tempo_inference(
    data_root,
    model_checkpoint,
    output_dir,
    device='cuda'
):
    """
    Run TEMPO inference on your CHC data using the pretrained Vimeo model.
    
    Args:
        data_root: Path to your tempo_format dataset (created by previous script)
        model_checkpoint: Path to pretrained TEMPO model
        output_dir: Where to save interpolated frames
        device: 'cuda' or 'cpu'
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("TEMPO Inference on CHC Data")
    print("="*60)
    
    # Load dataset using existing Vimeo90K loader
    print(f"\n📂 Loading dataset from: {data_root}")
    dataset = Vimeo90KTriplet(
        root=data_root,
        split="test",
        mode="interp",
        crop_size=None,  # Full resolution (512x512)
        center_crop_eval=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=vimeo_collate
    )
    
    print(f"   Found {len(dataset)} triplets to process")
    
    # Load pretrained model
    print(f"\n🤖 Loading pretrained TEMPO model from: {model_checkpoint}")
    model = build_tempo(base_channels=64, temporal_channels=64)
    
    checkpoint = torch.load(model_checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state', checkpoint)
    
    # Handle DDP checkpoint (remove 'module.' prefix)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("   ✅ Model loaded successfully!")
    
    # Run inference
    print(f"\n🚀 Running inference...")
    print(f"   Output directory: {output_dir}")
    
    results = []
    
    with torch.no_grad():
        for idx, (frames, anchor_times, target_time, target) in enumerate(tqdm(dataloader, desc="Processing")):
            # Move to device
            frames = frames.to(device)
            anchor_times = anchor_times.to(device)
            target_time = target_time.to(device)
            target = target.to(device)
            
            # Generate interpolated frame
            output, aux = model(frames, anchor_times, target_time)
            
            # Clamp to [0, 1]
            output = output.clamp(0, 1)
            
            # Calculate metrics (PSNR, SSIM)
            mse = torch.nn.functional.mse_loss(output, target)
            psnr = -10 * torch.log10(mse + 1e-8)
            
            # Save interpolated frame
            output_img = output[0].cpu()  # (3, H, W)
            
            # Convert RGB back to grayscale (take mean of channels)
            grayscale = output_img.mean(dim=0).numpy()  # (H, W)
            
            # Scale to uint8 [0, 255]
            img_uint8 = (grayscale * 255).astype(np.uint8)
            
            # Save
            save_path = output_path / f"interpolated_frame_{idx+1:04d}.png"
            cv2.imwrite(str(save_path), img_uint8)
            
            # Also save ground truth and inputs for comparison
            if idx == 0:  # Save first triplet for visual inspection
                comparison_dir = output_path / "comparison"
                comparison_dir.mkdir(exist_ok=True)
                
                # Save input frames
                frame1 = frames[0, 0].cpu().mean(dim=0).numpy()
                frame2 = frames[0, 1].cpu().mean(dim=0).numpy()
                cv2.imwrite(str(comparison_dir / "input_frame1.png"), (frame1 * 255).astype(np.uint8))
                cv2.imwrite(str(comparison_dir / "input_frame3.png"), (frame2 * 255).astype(np.uint8))
                
                # Save ground truth
                gt = target[0].cpu().mean(dim=0).numpy()
                cv2.imwrite(str(comparison_dir / "ground_truth_frame2.png"), (gt * 255).astype(np.uint8))
                
                # Save interpolated
                cv2.imwrite(str(comparison_dir / "interpolated_frame2.png"), img_uint8)
            
            results.append({
                'triplet': idx,
                'psnr': psnr.item(),
                'confidence': aux['conf_map'].mean().item()
            })
    
    # Print summary
    print(f"\n" + "="*60)
    print("✅ Inference Complete!")
    print("="*60)
    print(f"   Processed: {len(results)} frames")
    print(f"   Average PSNR: {np.mean([r['psnr'] for r in results]):.2f} dB")
    print(f"   Output saved to: {output_dir}")
    print(f"   Comparison samples: {output_dir}/comparison/")
    
    # Save metrics to CSV
    import csv
    metrics_file = output_path / "metrics.csv"
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['triplet', 'psnr', 'confidence'])
        writer.writeheader()
        writer.writerows(results)
    print(f"   Metrics saved to: {metrics_file}")
    
    return results


if __name__ == "__main__":
    # Paths
    data_root = "/home/groups/ChangLab/govindsa/confocal_project/datasets/tempo_format/input/H4_Q3_3_ch0_triplet"
    #### Trained on vimeo data
    # model_checkpoint = "/home/groups/ChangLab/govindsa/confocal_project/TEMPO/code/TEMPO/runs/2025_10_25_11_19_08/checkpoints/best_model.pth"
    #### Trained on Atlas specimen data
    model_checkpoint = "/home/groups/ChangLab/govindsa/confocal_project/TEMPO/code/TEMPO/runs/training_atlas_specimen_on_channel3_11_09/checkpoints/best_model.pth"
    #### Trained on Vimeo
    # output_dir = "/home/groups/ChangLab/govindsa/confocal_project/datasets/tempo_format/output/2025_10_25_11_19_08/H4_Q3_3_ch4_interpolated"
    output_dir = "/home/groups/ChangLab/govindsa/confocal_project/datasets/tempo_format/output/training_atlas_specimen_on_channel3_11_09/H4_Q3_3_ch0_interpolated"
    # Run inference
    results = run_tempo_inference(
        data_root=data_root,
        model_checkpoint=model_checkpoint,
        output_dir=output_dir,
        device='cuda'
    )
    
    print("\n🎉 Done!")