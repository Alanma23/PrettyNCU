import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export interface Diagnostic {
  id: string;
  name: string;
  description: string;
  priority: string;
  overhead: string;
  flags: Record<string, string>;
  key_metrics: string[];
  when_to_use: string;
  llm_guidance: string;
}

export interface Workflow {
  name: string;
  description: string;
  steps: Array<{
    diagnostic?: string;
    variant?: string;
    decision?: string;
    next?: string;
    note?: string;
  }>;
}

export interface DiagnosticsDatabase {
  diagnostics: Diagnostic[];
  workflows: Workflow[];
  thresholds: Record<string, any>;
  compilation_requirements: Record<string, any>;
}

export class DiagnosticsCompiler {
  private database: DiagnosticsDatabase;
  private databasePath: string;

  constructor(databasePath?: string) {
    this.databasePath = databasePath || join(__dirname, '../../diagnostics-database.json');
    this.database = this.loadDatabase();
  }

  /**
   * Load the diagnostics database from JSON
   */
  private loadDatabase(): DiagnosticsDatabase {
    try {
      const content = readFileSync(this.databasePath, 'utf-8');
      return JSON.parse(content);
    } catch (error) {
      throw new Error(`Failed to load diagnostics database from ${this.databasePath}: ${error}`);
    }
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
   * Get all available workflows
   */
  getAllWorkflows(): Workflow[] {
    return this.database.workflows;
  }

  /**
   * Compile selected diagnostics into LLM context string
   */
  compileContext(diagnosticIds: string[]): string {
    if (diagnosticIds.length === 0) {
      return this.compileDefaultContext();
    }

    const selectedDiagnostics = diagnosticIds
      .map(id => this.getDiagnostic(id))
      .filter(d => d !== undefined) as Diagnostic[];

    if (selectedDiagnostics.length === 0) {
      return this.compileDefaultContext();
    }

    return this.formatDiagnosticsContext(selectedDiagnostics);
  }

  /**
   * Compile a workflow into LLM context
   */
  compileWorkflowContext(workflowName: string): string {
    const workflow = this.database.workflows.find(w => w.name === workflowName);
    if (!workflow) {
      return '';
    }

    const diagnosticIds = new Set<string>();
    for (const step of workflow.steps) {
      if (step.diagnostic) {
        diagnosticIds.add(step.diagnostic);
      }
    }

    const diagnostics = Array.from(diagnosticIds)
      .map(id => this.getDiagnostic(id))
      .filter(d => d !== undefined) as Diagnostic[];

    let context = `WORKFLOW: ${workflow.name}\n`;
    context += `${workflow.description}\n\n`;
    context += this.formatWorkflow(workflow);
    context += '\n\n';
    context += this.formatDiagnosticsContext(diagnostics);

    return context;
  }

  /**
   * Format diagnostics into LLM context string
   */
  private formatDiagnosticsContext(diagnostics: Diagnostic[]): string {
    let context = 'NCU DIAGNOSTIC CONFIGURATION\n';
    context += '='.repeat(80) + '\n\n';

    for (const diag of diagnostics) {
      context += `[${diag.id.toUpperCase()}] ${diag.name}\n`;
      context += `-`.repeat(80) + '\n';
      context += `Description: ${diag.description}\n`;
      context += `Priority: ${diag.priority.toUpperCase()} | Overhead: ${diag.overhead.toUpperCase()}\n\n`;

      context += `When to use:\n  ${diag.when_to_use}\n\n`;

      context += `LLM Guidance:\n  ${diag.llm_guidance}\n\n`;

      context += `NCU Flag Options:\n`;
      for (const [variant, flags] of Object.entries(diag.flags)) {
        context += `  ${variant}: ncu ${flags}\n`;
      }
      context += '\n';

      if (diag.key_metrics.length > 0) {
        context += `Key Metrics to Analyze:\n`;
        for (const metric of diag.key_metrics) {
          context += `  - ${metric}\n`;
        }
        context += '\n';
      }

      context += '\n';
    }

    // Add thresholds
    context += 'PERFORMANCE THRESHOLDS\n';
    context += '='.repeat(80) + '\n';
    for (const [name, threshold] of Object.entries(this.database.thresholds)) {
      context += `${name}: ${threshold.value}${threshold.unit} → ${threshold.action}\n`;
    }
    context += '\n';

    return context;
  }

  /**
   * Format a workflow for LLM context
   */
  private formatWorkflow(workflow: Workflow): string {
    let context = `Steps:\n`;

    for (let i = 0; i < workflow.steps.length; i++) {
      const step = workflow.steps[i];

      if (step.diagnostic) {
        const diag = this.getDiagnostic(step.diagnostic);
        context += `  ${i + 1}. Run: ${diag?.name || step.diagnostic}`;
        if (step.variant) {
          context += ` (${step.variant})`;
        }
        if (step.note) {
          context += ` - ${step.note}`;
        }
        context += '\n';
      } else if (step.decision) {
        context += `  ${i + 1}. Decision: ${step.decision}`;
        if (step.next) {
          context += ` → ${step.next}`;
        }
        context += '\n';
      }
    }

    return context;
  }

  /**
   * Compile default context (quick diagnostic workflow)
   */
  private compileDefaultContext(): string {
    return this.compileWorkflowContext('Initial Profiling');
  }

  /**
   * Get a list of available diagnostic IDs and names
   */
  getDiagnosticList(): Array<{ id: string; name: string; priority: string }> {
    return this.database.diagnostics.map(d => ({
      id: d.id,
      name: d.name,
      priority: d.priority
    }));
  }

  /**
   * Get a list of available workflow names
   */
  getWorkflowList(): Array<{ name: string; description: string }> {
    return this.database.workflows.map(w => ({
      name: w.name,
      description: w.description
    }));
  }
}
