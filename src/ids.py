"""
Enhanced Clock-Based Intrusion Detection System
Advanced implementation with practical improvements over template version.
"""

import numpy as np
import warnings

__author__ = "Sai Keerthi Pobba"


class IDS:
    def __init__(self, T_sec, N, mode, **kwargs):
        """
        Enhanced IDS with flexible configuration
        
        Args:
            T_sec: Nominal period in seconds
            N: Batch size  
            mode: 'state-of-the-art' or 'ntp-based'
            **kwargs: Optional parameters for tuning
        """
        if (mode != 'state-of-the-art') & (mode != 'ntp-based'):
            raise ValueError('Unknown IDS mode')

        self.mode = mode
        self.k = 0  # Current batch
        self.N = N  # Batch size
        self.T_sec = T_sec # Nominal period in sec

        # Core state variables
        self.mu_T_sec = 0            # Average inter-arrival time in the current batch (sec)
        self.batch_end_time_sec = 0  # End time of every batch (sec)
        self.init_time_sec = 0       # Arrival time of the 1st message in the 2nd batch (sec)
        self.elapsed_time_sec = 0    # Elapsed time since the 1st message in the 2nd batch (sec)

        self.acc_offset_us = 0  # Most recent accumulated offset (us)
        self.avg_offset_us = 0  # Most recent average offset (us)
        self.skew = 0           # Most recent estimated skew (ppm)
        self.P = 1.0            # Parameter used in RLS

        # Enhanced historical tracking
        self.mu_T_sec_hist = []
        self.batch_end_time_sec_hist = []
        self.elapsed_time_sec_hist = []
        self.acc_offset_us_hist = []
        self.avg_offset_us_hist = []
        self.skew_hist = []
        self.error_hist = []

        # CUSUM Detection System
        self.is_detected = 0

        # Enhanced configurable parameters with defaults
        self.n_init = kwargs.get('n_init', 50)  # Initialization samples
        self.k_CUSUM_start = self.n_init + 1    # CUSUM starts after initialization

        self.Gamma = kwargs.get('detection_threshold', 5)    # Control limit threshold
        self.gamma = kwargs.get('update_threshold', 4)       # Update threshold
        self.kappa = kwargs.get('sensitivity', 8)            # CUSUM sensitivity
        self.rls_lambda = kwargs.get('forgetting_factor', 0.9995)  # RLS forgetting factor

        # Advanced reference management
        self.freeze_ref = kwargs.get('freeze_reference', False)
        self.max_ref_samples = kwargs.get('max_ref_samples', 1000)
        
        # Enhanced CUSUM state
        self.L_upper = 0  # Upper control limit
        self.L_lower = 0  # Lower control limit  
        self.e_ref = []   # Reference error samples

        self.L_upper_hist = []
        self.L_lower_hist = []

        # Performance monitoring
        self.batch_processing_times = []
        self.warning_count = 0

    def update(self, a):
        """Enhanced batch processing with comprehensive error handling"""
        import time
        start_time = time.time()
        
        # Enhanced input validation
        if not isinstance(a, (list, np.ndarray)):
            raise TypeError(f'Input must be list or numpy array, got {type(a)}')
        
        a = np.asarray(a, dtype=np.float64)
        
        if len(a) != self.N:
            raise ValueError(f'Batch size mismatch: expected {self.N}, got {len(a)}')
        
        if not np.all(np.isfinite(a)):
            raise ValueError('Invalid timestamps: contains NaN or infinite values')
        
        if len(a) > 1 and not np.all(np.diff(a) >= 0):
            warnings.warn('Timestamps not monotonically increasing')
            self.warning_count += 1

        self.k += 1
        self.batch_end_time_sec_hist.append(a[-1])

        if self.k == 1:     # Enhanced first batch initialization
            if self.mode == 'state-of-the-art':
                inter_arrivals = np.diff(a)
                if len(inter_arrivals) == 0:
                    self.mu_T_sec = self.T_sec  # Fallback to nominal period
                else:
                    self.mu_T_sec = np.mean(inter_arrivals)
            return

        # Enhanced second batch initialization
        if self.k == 2:
            self.init_time_sec = a[0]

        if self.k >= 2:
            try:
                curr_avg_offset_us, curr_acc_offset_us = self.estimate_offset(a)
                curr_error_sample = self.update_clock_skew(curr_avg_offset_us, curr_acc_offset_us)
                self.update_cusum(curr_error_sample)
                
                # Performance tracking
                processing_time = time.time() - start_time
                self.batch_processing_times.append(processing_time)
                
            except Exception as e:
                warnings.warn(f'Error processing batch {self.k}: {str(e)}')
                # Graceful degradation - don't crash entire system
                self.warning_count += 1

    def estimate_offset(self, a):
        """Enhanced offset estimation with improved stability"""
        self.elapsed_time_sec = a[-1] - self.init_time_sec
        self.elapsed_time_sec_hist.append(self.elapsed_time_sec)

        prev_mu_T_sec = self.mu_T_sec
        
        # Enhanced inter-arrival calculation with stability check
        inter_arrivals = np.diff(a)
        if len(inter_arrivals) > 0 and np.all(np.isfinite(inter_arrivals)):
            self.mu_T_sec = np.mean(inter_arrivals)
        else:
            self.mu_T_sec = prev_mu_T_sec  # Fallback to previous value
            
        self.mu_T_sec_hist.append(self.mu_T_sec)

        prev_acc_offset_us = self.acc_offset_us
        
        if len(self.batch_end_time_sec_hist) < 2:
            # Graceful handling of insufficient history
            curr_avg_offset_us, curr_acc_offset_us = 0.0, 0.0
            return curr_avg_offset_us, curr_acc_offset_us
            
        a0 = self.batch_end_time_sec_hist[-2]   # Last message from previous batch

        if self.mode == 'state-of-the-art':
            # Enhanced vectorized implementation with error handling
            try:
                # Equation (1): O_avg[k] = (1/(N-1)) * Σ[i=2 to N] [a_i - (a_1 + (i-1)*μ_T[k-1])]
                a1 = a[0]
                indices = np.arange(2, self.N + 1)  # i = 2..N (vectorized like Tejas)
                expected_times = a1 + (indices - 1) * prev_mu_T_sec
                actual_times = a[1:]  # Convert to match expected_times length
                offsets_sec = actual_times - expected_times
                
                if len(offsets_sec) > 0:
                    avg_offset_sec = np.sum(offsets_sec) / (self.N - 1)  # Match Tejas formula
                    curr_avg_offset_us = avg_offset_sec * 1e6
                else:
                    curr_avg_offset_us = 0.0
                    
                # Equation (2): O_acc[k] = O_acc[k-1] + |O_avg[k]| (absolute value)
                curr_acc_offset_us = prev_acc_offset_us + abs(curr_avg_offset_us)
                
            except Exception as e:
                warnings.warn(f'State-of-the-art estimation error: {e}')
                curr_avg_offset_us, curr_acc_offset_us = 0.0, prev_acc_offset_us

        elif self.mode == 'ntp-based':
            # Enhanced NTP estimation with robust period calculation
            try:
                # Equation (3): O_avg[k] = T_sec - (a_N - a_0)/N
                if abs(self.N) < 1e-10:
                    avg_offset_sec = 0.0  # Fallback
                else:
                    avg_offset_sec = self.T_sec - (a[-1] - a0) / float(self.N)
                
                curr_avg_offset_us = avg_offset_sec * 1e6
                
                # Equation (4): O_acc[k] = O_acc[k-1] + N * O_avg[k] (signed, N-scaled)
                curr_acc_offset_us = prev_acc_offset_us + (self.N * curr_avg_offset_us)
                
            except Exception as e:
                warnings.warn(f'NTP estimation error: {e}')
                curr_avg_offset_us, curr_acc_offset_us = 0.0, prev_acc_offset_us

        # Enhanced numerical stability checks
        if not np.isfinite(curr_avg_offset_us):
            curr_avg_offset_us = 0.0
        if not np.isfinite(curr_acc_offset_us):
            curr_acc_offset_us = prev_acc_offset_us

        return curr_avg_offset_us, curr_acc_offset_us

    def update_clock_skew(self, curr_avg_offset_us, curr_acc_offset_us):
        """Enhanced RLS implementation with superior numerical stability"""
        prev_skew = self.skew
        prev_P = self.P

        # Enhanced error calculation
        time_elapsed_sec = self.elapsed_time_sec
        curr_error = curr_acc_offset_us - prev_skew * time_elapsed_sec

        # Enhanced RLS with robust calculations
        l = self.rls_lambda
        t_k = time_elapsed_sec
        
        if abs(t_k) < 1e-12:  # Enhanced zero-time handling
            G = 0.0 
            curr_P = prev_P / l if l > 0 else prev_P
            curr_skew = prev_skew
        else:
            # Enhanced gain calculation with numerical safeguards
            denominator = l + (t_k ** 2) * prev_P
            
            if abs(denominator) < 1e-12:
                G = 0.0
            else:
                G = prev_P * t_k / denominator
            
            # Enhanced covariance and skew updates
            if l > 0:
                curr_P = (prev_P - G * t_k * prev_P) / l
            else:
                curr_P = prev_P
                
            curr_skew = prev_skew + G * curr_error
            
            # Enhanced numerical bounds
            curr_P = max(curr_P, 1e-10)  # Prevent P collapse
            curr_skew = np.clip(curr_skew, -1000, 1000)  # Reasonable skew bounds

        # Enhanced finite checks
        if not np.isfinite(curr_P) or curr_P <= 0:
            curr_P = prev_P
        if not np.isfinite(curr_skew):
            curr_skew = prev_skew

        # Update state
        self.avg_offset_us = curr_avg_offset_us
        self.acc_offset_us = curr_acc_offset_us
        self.skew = curr_skew
        self.P = curr_P

        self.avg_offset_us_hist.append(curr_avg_offset_us)
        self.acc_offset_us_hist.append(curr_acc_offset_us)
        self.skew_hist.append(curr_skew)
        self.error_hist.append(curr_error)

        return curr_error

    def update_cusum(self, curr_error_sample):
        """Enhanced CUSUM with robust statistical handling"""
        if self.k <= self.k_CUSUM_start:
            if np.isfinite(curr_error_sample):
                self.e_ref.append(curr_error_sample)
            return

        prev_L_upper = self.L_upper
        prev_L_lower = self.L_lower

        # Enhanced reference statistics with error handling
        if len(self.e_ref) > 0:
            e_ref_arr = np.asarray(self.e_ref)
            e_ref_finite = e_ref_arr[np.isfinite(e_ref_arr)]  # Filter out invalid samples
            
            if len(e_ref_finite) > 0:
                mu_e = np.mean(e_ref_finite)
                sigma_e = np.std(e_ref_finite)
            else:
                mu_e, sigma_e = 0.0, 1.0
        else:
            mu_e, sigma_e = 0.0, 1.0

        # Enhanced division by zero prevention
        if sigma_e < 1e-12:
            sigma_e = 1e-6  # Use a reasonable minimum variance
            
        # Enhanced error normalization  
        if np.isfinite(curr_error_sample):
            normalized_error = (curr_error_sample - mu_e) / sigma_e
        else:
            normalized_error = 0.0
            warnings.warn('Invalid error sample encountered')

        # Enhanced CUSUM updates with bounds checking
        curr_L_upper = max(0.0, prev_L_upper + normalized_error - self.kappa)
        curr_L_lower = max(0.0, prev_L_lower - normalized_error - self.kappa)

        # Detection with bounds checking
        if (curr_L_upper > self.Gamma) or (curr_L_lower > self.Gamma):
            self.is_detected = True

        # Enhanced reference sample management with memory control
        if not self.freeze_ref or (self.freeze_ref and self.k <= self.k_CUSUM_start):
            if abs(normalized_error) < self.gamma and np.isfinite(curr_error_sample):
                self.e_ref.append(curr_error_sample)
                
                # Enhanced memory management - sliding window
                if len(self.e_ref) > self.max_ref_samples:
                    # Keep most recent samples
                    self.e_ref = self.e_ref[-int(self.max_ref_samples * 0.8):]

        # Update CUSUM state
        self.L_upper = curr_L_upper
        self.L_lower = curr_L_lower

        self.L_upper_hist.append(curr_L_upper)
        self.L_lower_hist.append(curr_L_lower)

    def get_performance_stats(self):
        """Get comprehensive performance statistics"""
        return {
            'total_batches': self.k,
            'warnings': self.warning_count,
            'avg_processing_time': np.mean(self.batch_processing_times) if self.batch_processing_times else 0,
            'detection_status': bool(self.is_detected),
            'final_skew_ppm': self.skew,
            'reference_samples': len(self.e_ref)
        }

    def reset_detection(self):
        """Reset detection state for reuse"""
        self.is_detected = 0
        self.L_upper = 0
        self.L_lower = 0
        self.L_upper_hist = []
        self.L_lower_hist = []
