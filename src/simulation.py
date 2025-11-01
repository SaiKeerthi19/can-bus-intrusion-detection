from ids import *
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

__author__ = "Sai Keerthi Pobba"


# Configuration for flexible execution
CONFIG = {
    'batch_sizes': [20, 30],  # Easy to change N values
    'data_path': '../data',
    'results_path': '../results',
    'plot_style': {
        'figsize': (12, 8),
        'linewidth': 2,
        'fontsize': 14
    }
}


def import_data(file=None):
    """Enhanced data import with error handling"""
    try:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Data file not found: {file}")
            
        with open(file) as f:
            lines = f.readlines()
        
        data = []
        for i, line in enumerate(lines):
            try:
                value = float(line.strip())
                if np.isfinite(value):
                    data.append(value)
            except ValueError:
                warnings.warn(f"Invalid data at line {i+1} in {file}")
                continue
                
        if len(data) == 0:
            raise ValueError(f"No valid data found in {file}")
            
        return data
        
    except Exception as e:
        print(f"Error importing data from {file}: {e}")
        return []


def plot_acc_offsets(ids, mode):
    """Enhanced plotting with automatic N handling and error checking"""
    plt.figure(figsize=CONFIG['plot_style']['figsize'])
    
    try:
        if mode == 'state-of-the-art':
            # Enhanced state-of-the-art plotting with error handling
            colors = ['blue', 'red', 'green']
            ecu_labels = ['ECU A (0x184)', 'ECU B (0x3d1)', 'ECU C (0x180)']
            ecu_keys = ['184-sota', '3d1-sota', '180-sota']
            
            for i, (key, label, color) in enumerate(zip(ecu_keys, ecu_labels, colors)):
                if key in ids and len(ids[key].elapsed_time_sec_hist) > 0:
                    time_data = np.array(ids[key].elapsed_time_sec_hist)
                    offset_data = np.array(ids[key].acc_offset_us_hist) * 1e-6  # Convert to seconds
                    
                    if len(time_data) == len(offset_data):
                        plt.plot(time_data, offset_data, color=color, 
                               label=f'{label} (skew: {ids[key].skew:.2f} ppm)', 
                               linewidth=CONFIG['plot_style']['linewidth'])
                    else:
                        warnings.warn(f'Data length mismatch for {key}')
            
            plt.title(f'Accumulated Offset vs Time (State-of-the-Art, N={ids["184-sota"].N})', 
                     fontsize=16, fontweight='bold')
            
        elif mode == 'ntp-based':
            # Enhanced NTP-based plotting with error handling
            colors = ['blue', 'red', 'green']
            ecu_labels = ['ECU A (0x184)', 'ECU B (0x3d1)', 'ECU C (0x180)']
            ecu_keys = ['184-ntp', '3d1-ntp', '180-ntp']
            
            for i, (key, label, color) in enumerate(zip(ecu_keys, ecu_labels, colors)):
                if key in ids and len(ids[key].elapsed_time_sec_hist) > 0:
                    time_data = np.array(ids[key].elapsed_time_sec_hist)
                    offset_data = np.array(ids[key].acc_offset_us_hist) * 1e-6  # Convert to seconds
                    
                    if len(time_data) == len(offset_data):
                        plt.plot(time_data, offset_data, color=color,
                               label=f'{label} (skew: {ids[key].skew:.2f} ppm)', 
                               linewidth=CONFIG['plot_style']['linewidth'])
                    else:
                        warnings.warn(f'Data length mismatch for {key}')
            
            plt.title(f'Accumulated Offset vs Time (NTP-based, N={ids["184-ntp"].N})', 
                     fontsize=16, fontweight='bold')

        plt.xlabel('Elapsed Time (s)', fontsize=CONFIG['plot_style']['fontsize'])
        plt.ylabel('Accumulated Offset (s)', fontsize=CONFIG['plot_style']['fontsize'])
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plots for Tasks 2, 3, 4
        os.makedirs(CONFIG['results_path'], exist_ok=True)
        if mode == 'state-of-the-art':
            N = ids['184-sota'].N
            filename = f"task2_accumulated_offsets_state_of_the_art_n{N}.png"
        elif mode == 'ntp-based':
            N = ids['184-ntp'].N
            filename = f"task3_accumulated_offsets_ntp_based_n{N}.png"
        
        filepath = os.path.join(CONFIG['results_path'], filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📁 Plot saved: {filename}")
        
    except Exception as e:
        print(f"Error creating plot for {mode}: {e}")


def create_attack_dataset(data_target, data_attacker, N, delta_T=0):
    """Enhanced dataset creation with timing manipulation support"""
    try:
        # Prepare data with enhanced error checking
        target_samples = 1000 * N
        attacker_samples = 1000 * N
        
        if len(data_target) < target_samples:
            warnings.warn(f"Insufficient target data: need {target_samples}, got {len(data_target)}")
            target_samples = len(data_target)
            
        if len(data_attacker) < attacker_samples:
            warnings.warn(f"Insufficient attacker data: need {attacker_samples}, got {len(data_attacker)}")
            attacker_samples = len(data_attacker)

        # Create relative timestamps
        data_target_rel = np.asarray(data_target[0:target_samples]) - data_target[0]
        data_attacker_rel = np.asarray(data_attacker[0:attacker_samples]) - data_attacker[0]
        
        # Apply timing manipulation if specified (for cloaking attacks)
        if abs(delta_T) > 0:
            manipulated_data = []
            for i in range(0, len(data_attacker_rel), N):
                batch = data_attacker_rel[i:i+N]
                if len(batch) == N:
                    # Apply progressive timing adjustment within batch
                    for j in range(len(batch)):
                        batch[j] += j * delta_T
                    manipulated_data.extend(batch)
            data_attacker_rel = np.asarray(manipulated_data)
        
        # Combine datasets: legitimate + attack (with proper timing)
        if len(data_target_rel) > 0 and len(data_attacker_rel) > 0:
            combined_data = np.append(data_target_rel, data_target_rel[-1] + 0.1 + data_attacker_rel)
        else:
            raise ValueError("Insufficient data for attack simulation")
        
        return combined_data
        
    except Exception as e:
        print(f"Error creating attack dataset: {e}")
        return np.array([])


def simulation_masquerade_attack(mode):
    """Enhanced masquerade attack simulation with flexible N support"""
    plt.figure(figsize=CONFIG['plot_style']['figsize'])
    
    try:
        data_184 = import_data('../data/184.txt')
        data_3d1 = import_data('../data/3d1.txt')
        
        if len(data_184) == 0 or len(data_3d1) == 0:
            print(f"Error: Cannot load data for masquerade attack ({mode})")
            return

        N = CONFIG['batch_sizes'][0]  # Use first configured batch size
        
        # Create attack dataset
        data = create_attack_dataset(data_184, data_3d1, N)
        
        if len(data) == 0:
            return

        ids = IDS(T_sec=0.1, N=N, mode=mode)
        
        batch_num = min(2000, len(data) // N)  # Adaptive batch number
        attack_start_cusum = 1000 - ids.k_CUSUM_start  # Account for CUSUM delay
        
        for i in range(batch_num):
            if (i + 1) * N <= len(data):
                batch = np.asarray(data[i * N:(i + 1) * N])
                ids.update(batch)

        # Enhanced plotting with detection indication
        plt.plot(ids.L_upper_hist, 'b-', label='L+ (Upper Control Limit)', 
                linewidth=CONFIG['plot_style']['linewidth'])
        plt.plot(ids.L_lower_hist, 'r-', label='L- (Lower Control Limit)', 
                linewidth=CONFIG['plot_style']['linewidth'])
        plt.axhline(y=ids.Gamma, color='black', linestyle='--', 
                   label=f'Detection Threshold (Γ={ids.Gamma})', 
                   linewidth=CONFIG['plot_style']['linewidth'])
        
        if attack_start_cusum > 0 and attack_start_cusum < len(ids.L_upper_hist):
            plt.axvline(x=attack_start_cusum, color='orange', linestyle=':', 
                       label='Attack Start (Batch 1001)', 
                       linewidth=CONFIG['plot_style']['linewidth'])
        
        # Mark detection point if detected
        if ids.is_detected:
            detection_point = None
            for i, (L_u, L_l) in enumerate(zip(ids.L_upper_hist, ids.L_lower_hist)):
                if L_u > ids.Gamma or L_l > ids.Gamma:
                    detection_point = i
                    break
            if detection_point is not None:
                plt.axvline(x=detection_point, color='red', linestyle='-',
                           label=f'Attack Detected (Batch {detection_point + ids.k_CUSUM_start + 1})',
                           linewidth=CONFIG['plot_style']['linewidth'])

        plt.xlabel('Batch Number (from CUSUM start)', fontsize=CONFIG['plot_style']['fontsize'])
        plt.ylabel('Control Limit Value', fontsize=CONFIG['plot_style']['fontsize'])
        plt.title(f'Masquerade Attack Detection ({mode.title()} IDS, N={N})', 
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save results
        os.makedirs(CONFIG['results_path'], exist_ok=True)
        filename = f"task5_masquerade_{mode.replace('-', '_')}_n{N}.png"
        plt.savefig(os.path.join(CONFIG['results_path'], filename), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ {mode.title()} masquerade attack: {'Detected' if ids.is_detected else 'Not detected'}")
        
    except Exception as e:
        print(f"Error in masquerade attack simulation ({mode}): {e}")


def simulation_cloaking_attack(mode):
    """Enhanced cloaking attack simulation with timing manipulation"""
    plt.figure(figsize=CONFIG['plot_style']['figsize'])
    
    try:
        data_184 = import_data('../data/184.txt')
        data_180 = import_data('../data/180.txt')
        
        if len(data_184) == 0 or len(data_180) == 0:
            print(f"Error: Cannot load data for cloaking attack ({mode})")
            return

        N = CONFIG['batch_sizes'][0]  # Use first configured batch size
        delta_T = -29e-6  # Cloaking timing manipulation
        
        # Create attack dataset with timing manipulation
        data = create_attack_dataset(data_184, data_180, N, delta_T)
        
        if len(data) == 0:
            return

        ids = IDS(T_sec=0.1, N=N, mode=mode)
        
        batch_num = min(2000, len(data) // N)  # Adaptive batch number
        attack_start_cusum = 1000 - ids.k_CUSUM_start  # Account for CUSUM delay
        
        for i in range(batch_num):
            if (i + 1) * N <= len(data):
                batch = np.asarray(data[i * N:(i + 1) * N])
                ids.update(batch)

        # Enhanced plotting
        plt.plot(ids.L_upper_hist, 'b-', label='L+ (Upper Control Limit)', 
                linewidth=CONFIG['plot_style']['linewidth'])
        plt.plot(ids.L_lower_hist, 'r-', label='L- (Lower Control Limit)', 
                linewidth=CONFIG['plot_style']['linewidth'])
        plt.axhline(y=ids.Gamma, color='black', linestyle='--', 
                   label=f'Detection Threshold (Γ={ids.Gamma})', 
                   linewidth=CONFIG['plot_style']['linewidth'])
        
        if attack_start_cusum > 0 and attack_start_cusum < len(ids.L_upper_hist):
            plt.axvline(x=attack_start_cusum, color='orange', linestyle=':', 
                       label='Attack Start (Batch 1001)', 
                       linewidth=CONFIG['plot_style']['linewidth'])

        # Mark detection point if detected
        if ids.is_detected:
            detection_point = None
            for i, (L_u, L_l) in enumerate(zip(ids.L_upper_hist, ids.L_lower_hist)):
                if L_u > ids.Gamma or L_l > ids.Gamma:
                    detection_point = i
                    break
            if detection_point is not None:
                plt.axvline(x=detection_point, color='red', linestyle='-',
                           label=f'Attack Detected (Batch {detection_point + ids.k_CUSUM_start + 1})',
                           linewidth=CONFIG['plot_style']['linewidth'])

        plt.xlabel('Batch Number (from CUSUM start)', fontsize=CONFIG['plot_style']['fontsize'])
        plt.ylabel('Control Limit Value', fontsize=CONFIG['plot_style']['fontsize'])
        plt.title(f'Cloaking Attack Detection ({mode.title()} IDS, N={N})\n(ΔT = -29μs per message)', 
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save results
        os.makedirs(CONFIG['results_path'], exist_ok=True)
        filename = f"task6_cloaking_{mode.replace('-', '_')}_n{N}.png"
        plt.savefig(os.path.join(CONFIG['results_path'], filename), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ {mode.title()} cloaking attack: {'Detected' if ids.is_detected else 'Not detected'}")
        
    except Exception as e:
        print(f"Error in cloaking attack simulation ({mode}): {e}")


def run_analysis_for_batch_size(N):
    """Run complete analysis for specified batch size"""
    print(f"\n{'='*60}")
    print(f"RUNNING ANALYSIS FOR BATCH SIZE N = {N}")
    print(f"{'='*60}")
    
    try:
        # Load and validate data
        data_184 = import_data('../data/184.txt')
        data_3d1 = import_data('../data/3d1.txt')
        data_180 = import_data('../data/180.txt')

        if not all([data_184, data_3d1, data_180]):
            print("Error: Could not load all required data files")
            return

        # Convert to relative timestamps
        data_184 = np.asarray(data_184) - data_184[0]
        data_3d1 = np.asarray(data_3d1) - data_3d1[0]
        data_180 = np.asarray(data_180) - data_180[0]

        # Create IDS instances with enhanced configuration
        ids = {}
        ids['184-sota'] = IDS(T_sec=0.1, N=N, mode='state-of-the-art')
        ids['184-ntp'] = IDS(T_sec=0.1, N=N, mode='ntp-based')
        ids['3d1-sota'] = IDS(T_sec=0.1, N=N, mode='state-of-the-art')
        ids['3d1-ntp'] = IDS(T_sec=0.1, N=N, mode='ntp-based')
        ids['180-sota'] = IDS(T_sec=0.1, N=N, mode='state-of-the-art')
        ids['180-ntp'] = IDS(T_sec=0.1, N=N, mode='ntp-based')

        # Determine batch count based on available data and N
        min_data_length = min(len(data_184), len(data_3d1), len(data_180))
        batch_num = min(6000, min_data_length // N) if N == 20 else min(4000, min_data_length // N)
        
        print(f"Processing {batch_num} batches for each ECU...")

        # Enhanced batch processing with progress indication
        for i in range(batch_num):
            try:
                # Process ECU A (0x184)
                if (i + 1) * N <= len(data_184):
                    batch_184 = data_184[i*N:(i+1)*N]
                    ids['184-sota'].update(batch_184)
                    ids['184-ntp'].update(batch_184)

                # Process ECU B (0x3d1)
                if (i + 1) * N <= len(data_3d1):
                    batch_3d1 = data_3d1[i*N:(i+1)*N]
                    ids['3d1-sota'].update(batch_3d1)
                    ids['3d1-ntp'].update(batch_3d1)

                # Process ECU C (0x180)
                if (i + 1) * N <= len(data_180):
                    batch_180 = data_180[i*N:(i+1)*N]
                    ids['180-sota'].update(batch_180)
                    ids['180-ntp'].update(batch_180)
                    
            except Exception as e:
                warnings.warn(f"Error processing batch {i}: {e}")
                continue

        # Display performance statistics
        print("\nPerformance Summary:")
        for key, instance in ids.items():
            stats = instance.get_performance_stats()
            print(f"  {key}: {stats['total_batches']} batches, "
                  f"{stats['warnings']} warnings, "
                  f"skew: {stats['final_skew_ppm']:.2f} ppm")

        # Execute all tasks
        print(f"\n📊 Task 2: Plotting accumulated offsets (State-of-the-Art, N={N})")
        plot_acc_offsets(ids, "state-of-the-art")

        print(f"\n📊 Task 3: Plotting accumulated offsets (NTP-based, N={N})")
        plot_acc_offsets(ids, "ntp-based")

        print(f"\n🚨 Task 5: Simulating masquerade attacks (N={N})")
        simulation_masquerade_attack("state-of-the-art")
        simulation_masquerade_attack("ntp-based")

        print(f"\n🕵️ Task 6: Simulating cloaking attacks (N={N})")
        simulation_cloaking_attack("state-of-the-art")
        simulation_cloaking_attack("ntp-based")
        
    except Exception as e:
        print(f"Error in analysis for N={N}: {e}")


if __name__ == '__main__':
    """
    Enhanced main execution with flexible batch size configuration
    
    To change batch sizes, modify CONFIG['batch_sizes'] at the top of this file
    """
    print("EEP 595 Project 1: Clock Skew-Based Intrusion Detection System")
    print("Enhanced Implementation with Flexible Configuration")
    print("="*60)
    
    # Run analysis for each configured batch size
    for N in CONFIG['batch_sizes']:
        try:
            run_analysis_for_batch_size(N)
        except Exception as e:
            print(f"Failed to run analysis for N={N}: {e}")
    
    print(f"\n🎯 Analysis Complete for batch sizes: {CONFIG['batch_sizes']}")
    print(f"📁 Results saved to: {CONFIG['results_path']}")
