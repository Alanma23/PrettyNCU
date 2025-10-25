import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export interface DiagnosticFlags {
  [variant: string]: string;
}

export interface Diagnostic {
  id: string;
  name: string;
  description: string;
  priority: string;
  overhead: string;
  flags: DiagnosticFlags;
  key_metrics: string[];
  when_to_use: string;
  llm_guidance: string;
}

export interface Workflow {
  name: string;
  description: string;
  steps: any[];
}

export interface DiagnosticsDatabase {
  diagnostics: Diagnostic[];
  workflows: Workflow[];
  thresholds: any;
  compilation_requirements: any;
}

export interface DiagnosticSelection {
  id: string;
  variant: string;
  enabled: boolean;
  customFlags?: string;
}

export interface DiagnosticProfile {
  name: string;
  description: string;
  selections: DiagnosticSelection[];
  custom_context?: string;
}

export class DiagnosticsManager {
  private database: DiagnosticsDatabase;
  private selections: Map<string, DiagnosticSelection>;

  constructor(databasePath?: string) {
    const dbPath = databasePath || join(__dirname, '../../diagnostics-database.json');
    this.database = JSON.parse(readFileSync(dbPath, 'utf-8'));
    this.selections = new Map();

    // Initialize with default quick diagnostic enabled
    this.selections.set('quick_diagnostic', {
      id: 'quick_diagnostic',
      variant: 'quick',
      enabled: true
    });
  }

  /**
   * Get all available diagnostics
   */
  getAllDiagnostics(): Diagnostic[] {
    return this.database.diagnostics;
  }

  /**
   * Get a specific diagnostic by ID
   */
  getDiagnostic(id: string): Diagnostic | undefined {
    return this.database.diagnostics.find(d => d.id === id);
  }

  /**
   * Get diagnostics by priority
   */
  getDiagnosticsByPriority(priority: string): Diagnostic[] {
    return this.database.diagnostics.filter(d => d.priority === priority);
  }

  /**
   * Get all workflows
   */
  getWorkflows(): Workflow[] {
    return this.database.workflows;
  }

  /**
   * Set diagnostic selection
   */
  setDiagnostic(id: string, variant: string, enabled: boolean = true, customFlags?: string) {
    this.selections.set(id, {
      id,
      variant,
      enabled,
      customFlags
    });
  }

  /**
   * Toggle diagnostic on/off
   */
  toggleDiagnostic(id: string): boolean {
    const selection = this.selections.get(id);
    if (selection) {
      selection.enabled = !selection.enabled;
      return selection.enabled;
    }
    return false;
  }

  /**
   * Get current selections
   */
  getSelections(): DiagnosticSelection[] {
    return Array.from(this.selections.values()).filter(s => s.enabled);
  }

  /**
   * Clear all selections
   */
  clearSelections() {
    this.selections.clear();
  }

  /**
   * Apply a workflow (selects all diagnostics in the workflow)
   */
  applyWorkflow(workflowName: string) {
    const workflow = this.database.workflows.find(w => w.name === workflowName);
    if (!workflow) {
      throw new Error(`Workflow not found: ${workflowName}`);
    }

    this.clearSelections();

    for (const step of workflow.steps) {
      if (step.diagnostic) {
        this.setDiagnostic(step.diagnostic, step.variant || 'default', true);
      }
    }
  }

  /**
   * Compile selected diagnostics into LLM context
   */
  compileToContext(): string {
    const enabledDiagnostics = Array.from(this.selections.values()).filter(s => s.enabled);

    if (enabledDiagnostics.length === 0) {
      return 'NCU_DIAGNOSTICS: Use default quick profiling (--section SpeedOfLight)';
    }

    const contextParts: string[] = [];
    contextParts.push('=== NCU DIAGNOSTIC CONFIGURATION ===\n');

    for (const selection of enabledDiagnostics) {
      const diagnostic = this.getDiagnostic(selection.id);
      if (!diagnostic) continue;

      contextParts.push(`\n[${diagnostic.name}]`);
      contextParts.push(`Description: ${diagnostic.description}`);
      contextParts.push(`Priority: ${diagnostic.priority.toUpperCase()}`);
      contextParts.push(`Overhead: ${diagnostic.overhead}`);

      // Get the flags for this variant
      const flags = selection.customFlags || diagnostic.flags[selection.variant];
      if (flags) {
        contextParts.push(`NCU Command: ncu ${flags} ./executable`);
      }

      if (diagnostic.key_metrics.length > 0) {
        contextParts.push(`Key Metrics:`);
        diagnostic.key_metrics.forEach(m => contextParts.push(`  - ${m}`));
      }

      contextParts.push(`When to use: ${diagnostic.when_to_use}`);
      contextParts.push(`LLM Guidance: ${diagnostic.llm_guidance}`);
    }

    // Add thresholds
    contextParts.push('\n=== PERFORMANCE THRESHOLDS ===');
    contextParts.push(`Memory Throughput < ${this.database.thresholds.memory_throughput_low.value}% → Investigate latency issues`);
    contextParts.push(`Compute Throughput < ${this.database.thresholds.compute_throughput_low.value}% → Investigate latency issues`);
    contextParts.push(`Occupancy < ${this.database.thresholds.occupancy_low.value}% → Interferes with latency hiding`);
    contextParts.push(`L2 Miss Rate > ${this.database.thresholds.l2_miss_rate_high.value}% → Check coalescing or working set size`);

    // Add compilation requirements
    contextParts.push('\n=== COMPILATION REQUIREMENTS ===');
    contextParts.push(`For source correlation: Compile with ${this.database.compilation_requirements.source_correlation.flag}`);
    contextParts.push(`Required when: ${this.database.compilation_requirements.source_correlation.when}`);

    return contextParts.join('\n');
  }

  /**
   * Save current selection as a profile
   */
  saveProfile(name: string, description: string, profilePath: string) {
    const profile: DiagnosticProfile = {
      name,
      description,
      selections: Array.from(this.selections.values())
    };

    writeFileSync(profilePath, JSON.stringify(profile, null, 2), 'utf-8');
  }

  /**
   * Load a profile
   */
  loadProfile(profilePath: string) {
    if (!existsSync(profilePath)) {
      throw new Error(`Profile not found: ${profilePath}`);
    }

    const profile: DiagnosticProfile = JSON.parse(readFileSync(profilePath, 'utf-8'));

    this.clearSelections();

    for (const selection of profile.selections) {
      this.selections.set(selection.id, selection);
    }
  }

  /**
   * Get suggested diagnostics based on initial analysis hint
   */
  getSuggestedDiagnostics(hint: 'memory_bound' | 'compute_bound' | 'low_occupancy' | 'unknown'): Diagnostic[] {
    switch (hint) {
      case 'memory_bound':
        return this.database.diagnostics.filter(d =>
          ['memory_bottleneck', 'cache_control'].includes(d.id)
        );
      case 'compute_bound':
        return this.database.diagnostics.filter(d =>
          ['compute_workload', 'tensor_core'].includes(d.id)
        );
      case 'low_occupancy':
        return this.database.diagnostics.filter(d =>
          ['occupancy_roofline', 'warp_stalls'].includes(d.id)
        );
      case 'unknown':
      default:
        return this.database.diagnostics.filter(d =>
          d.id === 'quick_diagnostic'
        );
    }
  }

  /**
   * Generate a smart diagnostic plan based on code analysis
   */
  generateSmartPlan(codeHints: {
    hasSharedMemory?: boolean;
    hasTensorOps?: boolean;
    hasComplexBranching?: boolean;
    kernelSize?: 'small' | 'medium' | 'large';
  }): DiagnosticSelection[] {
    const plan: DiagnosticSelection[] = [];

    // Always start with quick diagnostic
    plan.push({
      id: 'quick_diagnostic',
      variant: 'quick',
      enabled: true
    });

    // Add tensor core check if tensor ops detected
    if (codeHints.hasTensorOps) {
      plan.push({
        id: 'tensor_core',
        variant: 'verify',
        enabled: true
      });
    }

    // Add occupancy check if shared memory detected
    if (codeHints.hasSharedMemory) {
      plan.push({
        id: 'occupancy_roofline',
        variant: 'occupancy_only',
        enabled: true
      });
    }

    // Add warp stall analysis if complex branching
    if (codeHints.hasComplexBranching) {
      plan.push({
        id: 'warp_stalls',
        variant: 'comprehensive',
        enabled: true
      });
    }

    return plan;
  }
}
