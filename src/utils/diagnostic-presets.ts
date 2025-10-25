import { DiagnosticsManager } from '../services/diagnostics-manager.js';

export type PresetName = 'quick' | 'memory' | 'compute' | 'occupancy' | 'latency' | 'tensor' | 'full';

/**
 * Apply a diagnostic preset and return the compiled context
 */
export function applyPreset(preset: PresetName): string {
  const manager = new DiagnosticsManager();

  switch (preset) {
    case 'quick':
      // Fastest - just SpeedOfLight
      manager.setDiagnostic('quick_diagnostic', 'quick', true);
      break;

    case 'memory':
      // Memory-bound investigation
      manager.setDiagnostic('quick_diagnostic', 'quick', true);
      manager.setDiagnostic('memory_bottleneck', 'focused', true);
      manager.setDiagnostic('memory_bottleneck', 'cache_analysis', true);
      break;

    case 'compute':
      // Compute-bound investigation
      manager.setDiagnostic('quick_diagnostic', 'quick', true);
      manager.setDiagnostic('compute_workload', 'combined', true);
      break;

    case 'occupancy':
      // Low occupancy investigation
      manager.setDiagnostic('quick_diagnostic', 'quick', true);
      manager.setDiagnostic('occupancy_roofline', 'occupancy_only', true);
      manager.setDiagnostic('warp_stalls', 'comprehensive', true);
      break;

    case 'latency':
      // Latency/stall analysis
      manager.setDiagnostic('quick_diagnostic', 'quick', true);
      manager.setDiagnostic('warp_stalls', 'comprehensive', true);
      manager.setDiagnostic('occupancy_roofline', 'occupancy_only', true);
      break;

    case 'tensor':
      // Tensor Core / GEMM optimization
      manager.setDiagnostic('quick_diagnostic', 'quick', true);
      manager.setDiagnostic('tensor_core', 'detailed', true);
      manager.setDiagnostic('memory_bottleneck', 'focused', true);
      break;

    case 'full':
      // Everything - very slow but comprehensive
      manager.setDiagnostic('quick_diagnostic', 'quick', true);
      manager.setDiagnostic('memory_bottleneck', 'comprehensive', true);
      manager.setDiagnostic('compute_workload', 'combined', true);
      manager.setDiagnostic('occupancy_roofline', 'full_roofline', true);
      manager.setDiagnostic('warp_stalls', 'comprehensive', true);
      break;

    default:
      throw new Error(`Unknown preset: ${preset}`);
  }

  return manager.compileToContext();
}

/**
 * Get description of a preset
 */
export function getPresetDescription(preset: PresetName): string {
  const descriptions: Record<PresetName, string> = {
    quick: 'Fast SpeedOfLight analysis (low overhead) - determines if memory or compute bound',
    memory: 'Memory bottleneck investigation - cache analysis and coalescing',
    compute: 'Compute workload analysis - pipeline utilization and instruction mix',
    occupancy: 'Low occupancy investigation - register/shared memory pressure',
    latency: 'Latency and warp stall analysis - pipeline stalls and scheduling',
    tensor: 'Tensor Core optimization - GEMM and matrix operations',
    full: 'Comprehensive analysis - all diagnostics (very high overhead)'
  };

  return descriptions[preset];
}

/**
 * List all available presets
 */
export function listPresets(): Array<{ name: PresetName; description: string }> {
  const presets: PresetName[] = ['quick', 'memory', 'compute', 'occupancy', 'latency', 'tensor', 'full'];

  return presets.map(name => ({
    name,
    description: getPresetDescription(name)
  }));
}
