"""
Sustainability and Environmental Impact Tracking
UCL COMP0220 Coursework - Carbon Footprint and Energy Consumption Metrics

This module tracks the environmental impact of model training and inference,
including:
- Energy consumption estimation
- CO2 emissions calculation
- FLOPS (Floating Point Operations) counting
- Training time and efficiency metrics
- Comparison with everyday energy equivalents
"""

import torch
import time
import psutil
import json
from typing import Dict, Optional
from datetime import datetime
import os


class SustainabilityTracker:
    """
    Track environmental impact of model training and inference
    
    Metrics tracked:
    - Energy consumption (kWh)
    - CO2 emissions (kg CO2)
    - Training time
    - GPU/CPU utilization
    - FLOPS
    """
    
    def __init__(self, model_name: str = "Model", track_gpu: bool = True):
        """
        Initialize sustainability tracker
        
        Args:
            model_name: Name of the model being tracked
            track_gpu: Whether to track GPU metrics (requires pynvml)
        """
        self.model_name = model_name
        self.track_gpu = track_gpu
        
        # Metrics storage
        self.start_time = None
        self.end_time = None
        self.training_times = []
        self.energy_consumed_kwh = 0.0
        
        # Constants
        self.CO2_PER_KWH_UK = 0.233  # kg CO2 per kWh (UK grid average 2023)
        self.CO2_PER_KWH_GLOBAL = 0.475  # kg CO2 per kWh (global average)
        
        # GPU tracking
        self.gpu_available = False
        if track_gpu and torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_available = True
                self.pynvml = pynvml
            except:
                print("⚠️  GPU tracking unavailable. Install pynvml for detailed GPU metrics.")
        
        # Session log
        self.session_log = {
            'model_name': model_name,
            'start_timestamp': None,
            'end_timestamp': None,
            'training_sessions': [],
            'total_energy_kwh': 0.0,
            'total_co2_kg': 0.0,
        }
    
    def start_training(self):
        """Start tracking a training session"""
        self.start_time = time.time()
        self.session_log['start_timestamp'] = datetime.now().isoformat()
        print(f"🌱 Sustainability tracking started for {self.model_name}")
    
    def end_training(self) -> Dict:
        """
        End tracking and calculate metrics
        
        Returns:
            Dictionary with sustainability metrics
        """
        self.end_time = time.time()
        self.session_log['end_timestamp'] = datetime.now().isoformat()
        
        training_time_hours = (self.end_time - self.start_time) / 3600.0
        self.training_times.append(training_time_hours)
        
        # Estimate energy consumption
        energy_kwh = self._estimate_energy_consumption(training_time_hours)
        self.energy_consumed_kwh += energy_kwh
        
        # Calculate CO2 emissions
        co2_kg_uk = energy_kwh * self.CO2_PER_KWH_UK
        co2_kg_global = energy_kwh * self.CO2_PER_KWH_GLOBAL
        
        # Record session
        session_data = {
            'duration_hours': training_time_hours,
            'energy_kwh': energy_kwh,
            'co2_kg_uk_grid': co2_kg_uk,
            'co2_kg_global_grid': co2_kg_global,
        }
        
        self.session_log['training_sessions'].append(session_data)
        self.session_log['total_energy_kwh'] = self.energy_consumed_kwh
        self.session_log['total_co2_kg'] = co2_kg_uk
        
        print(f"✅ Training session complete")
        print(f"   Duration: {training_time_hours:.2f} hours")
        print(f"   Energy: {energy_kwh:.3f} kWh")
        print(f"   CO2 (UK grid): {co2_kg_uk:.3f} kg")
        
        return session_data
    
    def _estimate_energy_consumption(self, hours: float) -> float:
        """
        Estimate energy consumption in kWh
        
        Args:
            hours: Training duration in hours
            
        Returns:
            Energy consumption in kWh
        """
        if self.gpu_available:
            # Get GPU power consumption
            try:
                power_mw = self.pynvml.nvmlDeviceGetPowerUsage(self.handle)
                power_w = power_mw / 1000.0  # Convert to watts
                
                # Add CPU and other components (~50W)
                total_power_w = power_w + 50
                
                # Calculate energy
                energy_kwh = (total_power_w * hours) / 1000.0
                
                return energy_kwh
            except:
                pass
        
        # Fallback: estimate based on typical hardware
        # RTX 3090: ~350W, RTX 4090: ~450W, CPU: ~65W, Other: ~50W
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            if '4090' in gpu_name or '4080' in gpu_name:
                estimated_power_w = 450 + 65 + 50  # High-end
            elif '3090' in gpu_name or '3080' in gpu_name:
                estimated_power_w = 350 + 65 + 50  # Mid-high
            else:
                estimated_power_w = 250 + 65 + 50  # Mid-range
        else:
            estimated_power_w = 65 + 50  # CPU only
        
        energy_kwh = (estimated_power_w * hours) / 1000.0
        return energy_kwh
    
    def get_carbon_footprint(self, grid: str = 'uk') -> Dict:
        """
        Calculate carbon footprint
        
        Args:
            grid: 'uk' or 'global' for grid carbon intensity
            
        Returns:
            Dictionary with carbon metrics and comparisons
        """
        co2_per_kwh = self.CO2_PER_KWH_UK if grid == 'uk' else self.CO2_PER_KWH_GLOBAL
        total_co2_kg = self.energy_consumed_kwh * co2_per_kwh
        
        # Interesting comparisons
        comparisons = self._get_carbon_comparisons(total_co2_kg)
        
        return {
            'total_energy_kwh': self.energy_consumed_kwh,
            'total_co2_kg': total_co2_kg,
            'grid_type': grid,
            'co2_per_kwh': co2_per_kwh,
            'comparisons': comparisons,
        }
    
    def _get_carbon_comparisons(self, co2_kg: float) -> Dict:
        """
        Get interesting comparisons for CO2 emissions
        
        Args:
            co2_kg: Total CO2 in kilograms
            
        Returns:
            Dictionary with various comparisons
        """
        return {
            # 1 mile driven in average car = 0.404 kg CO2
            'miles_driven': co2_kg / 0.404,
            
            # 1 tree absorbs ~21 kg CO2 per year
            'trees_to_offset_year': co2_kg / 21.0,
            
            # 1 smartphone charge = 0.008 kWh = ~0.002 kg CO2
            'smartphone_charges': co2_kg / 0.002,
            
            # 1 hour streaming video = ~0.05 kg CO2
            'hours_streaming': co2_kg / 0.05,
            
            # 1 Big Mac = ~2.5 kg CO2
            'big_macs': co2_kg / 2.5,
            
            # 1 kg of beef = ~27 kg CO2
            'kg_beef': co2_kg / 27.0,
        }
    
    def estimate_model_flops(self, model, input_shape: tuple) -> int:
        """
        Estimate FLOPS for a forward pass
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            
        Returns:
            Estimated FLOPS
        """
        # This is a simplified estimation
        # For accurate FLOPS, use libraries like thop or ptflops
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Rough estimate: 2 * params per forward pass
        # (multiply + add for each weight)
        flops_per_forward = total_params * 2
        
        return flops_per_forward
    
    def print_sustainability_report(self):
        """Print a comprehensive sustainability report"""
        print("\n" + "=" * 80)
        print(f"🌍 SUSTAINABILITY REPORT: {self.model_name}")
        print("=" * 80)
        
        if not self.training_times:
            print("No training sessions recorded yet.")
            return
        
        total_time_hours = sum(self.training_times)
        
        print(f"\n📊 Training Overview:")
        print(f"   Sessions: {len(self.training_times)}")
        print(f"   Total time: {total_time_hours:.2f} hours ({total_time_hours*60:.1f} minutes)")
        print(f"   Average session: {total_time_hours/len(self.training_times):.2f} hours")
        
        print(f"\n⚡ Energy Consumption:")
        print(f"   Total: {self.energy_consumed_kwh:.3f} kWh")
        print(f"   Cost (UK avg £0.34/kWh): £{self.energy_consumed_kwh * 0.34:.2f}")
        
        # CO2 emissions
        footprint_uk = self.get_carbon_footprint('uk')
        footprint_global = self.get_carbon_footprint('global')
        
        print(f"\n🌡️  Carbon Emissions:")
        print(f"   UK Grid: {footprint_uk['total_co2_kg']:.3f} kg CO2")
        print(f"   Global Grid: {footprint_global['total_co2_kg']:.3f} kg CO2")
        
        # Comparisons (using UK grid)
        comp = footprint_uk['comparisons']
        
        print(f"\n🔄 Equivalents (UK Grid):")
        print(f"   🚗 Miles driven: {comp['miles_driven']:.1f}")
        print(f"   🌳 Trees needed (1 year): {comp['trees_to_offset_year']:.2f}")
        print(f"   📱 Smartphone charges: {comp['smartphone_charges']:.0f}")
        print(f"   📺 Hours of streaming: {comp['hours_streaming']:.1f}")
        print(f"   🍔 Big Macs: {comp['big_macs']:.2f}")
        
        print(f"\n💡 Efficiency Recommendations:")
        if self.energy_consumed_kwh > 5.0:
            print("   ⚠️  High energy consumption detected")
            print("   → Consider: Model compression, quantization, or knowledge distillation")
            print("   → Use smaller batch sizes or fewer epochs if acceptable")
        else:
            print("   ✅ Reasonable energy consumption for this model")
        
        if not self.gpu_available:
            print("   💡 Install pynvml for accurate GPU power monitoring")
        
        print("\n" + "=" * 80)
    
    def save_report(self, filepath: str = "sustainability_report.json"):
        """Save sustainability report to JSON file"""
        report = {
            'model_name': self.model_name,
            'session_log': self.session_log,
            'uk_grid_footprint': self.get_carbon_footprint('uk'),
            'global_grid_footprint': self.get_carbon_footprint('global'),
            'total_training_hours': sum(self.training_times),
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Sustainability report saved: {filepath}")
    
    def compare_with_baseline(self, baseline_energy_kwh: float, baseline_accuracy: float,
                             current_accuracy: float):
        """
        Compare efficiency vs accuracy trade-off
        
        Args:
            baseline_energy_kwh: Energy consumed by baseline model
            baseline_accuracy: Baseline model accuracy
            current_accuracy: Current model accuracy
        """
        print("\n" + "=" * 80)
        print("⚖️  EFFICIENCY vs PERFORMANCE TRADE-OFF")
        print("=" * 80)
        
        energy_ratio = self.energy_consumed_kwh / baseline_energy_kwh if baseline_energy_kwh > 0 else 1.0
        accuracy_gain = current_accuracy - baseline_accuracy
        
        print(f"\n📊 Comparison:")
        print(f"   Baseline energy: {baseline_energy_kwh:.3f} kWh")
        print(f"   Current energy: {self.energy_consumed_kwh:.3f} kWh")
        print(f"   Energy ratio: {energy_ratio:.2f}x")
        print(f"\n   Baseline accuracy: {baseline_accuracy:.2%}")
        print(f"   Current accuracy: {current_accuracy:.2%}")
        print(f"   Accuracy gain: {accuracy_gain:+.2%}")
        
        # Efficiency score: accuracy gain per kWh
        if energy_ratio > 1.0:
            efficiency = accuracy_gain / (energy_ratio - 1.0) if energy_ratio > 1.0 else float('inf')
            print(f"\n💡 Efficiency Score: {efficiency:.4f} (accuracy gain per extra kWh ratio)")
            
            if accuracy_gain > 0 and energy_ratio < 2.0:
                print("   ✅ Good trade-off: Moderate energy increase for accuracy gain")
            elif accuracy_gain > 0.05 and energy_ratio < 5.0:
                print("   ⚠️  Acceptable: Significant energy increase but good accuracy gain")
            elif accuracy_gain < 0:
                print("   ❌ Poor trade-off: More energy but worse accuracy")
            else:
                print("   ⚠️  Questionable: High energy cost for small accuracy gain")
        
        print("=" * 80)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_training_carbon(hours: float, gpu_model: str = 'RTX3090', grid: str = 'uk') -> Dict:
    """
    Quick calculator for training carbon footprint
    
    Args:
        hours: Training duration in hours
        gpu_model: GPU model name
        grid: 'uk' or 'global'
        
    Returns:
        Dictionary with carbon estimates
    """
    # Power estimates (watts)
    gpu_power = {
        'RTX4090': 450,
        'RTX4080': 320,
        'RTX3090': 350,
        'RTX3080': 320,
        'RTX3070': 220,
        'V100': 300,
        'A100': 400,
    }
    
    power_w = gpu_power.get(gpu_model, 300) + 115  # GPU + CPU + other
    energy_kwh = (power_w * hours) / 1000.0
    
    co2_per_kwh = 0.233 if grid == 'uk' else 0.475
    co2_kg = energy_kwh * co2_per_kwh
    
    return {
        'hours': hours,
        'gpu_model': gpu_model,
        'power_watts': power_w,
        'energy_kwh': energy_kwh,
        'co2_kg': co2_kg,
        'grid': grid,
    }


if __name__ == "__main__":
    # Example usage
    print("Sustainability Tracker - Example")
    
    tracker = SustainabilityTracker("Example Model")
    tracker.start_training()
    
    # Simulate 2 hours of training
    import time
    print("Simulating training...")
    time.sleep(2)  # In reality, this would be your training loop
    
    tracker.end_training()
    tracker.print_sustainability_report()
    tracker.save_report()
