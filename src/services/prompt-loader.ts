import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export class PromptLoader {
  private promptsDir: string;

  constructor(promptsDir?: string) {
    // Default to prompts/ directory in project root
    this.promptsDir = promptsDir || join(__dirname, '../../prompts');
  }

  /**
   * Load the main system prompt
   */
  loadSystemPrompt(): string {
    return this.loadPromptFile('system-prompt.txt');
  }

  /**
   * Load NCU analyzer strategy prompt
   */
  loadNCUAnalyzer(): string {
    return this.loadPromptFile('ncu-analyzer.txt');
  }

  /**
   * Load code optimizer guidelines
   */
  loadCodeOptimizer(): string {
    return this.loadPromptFile('code-optimizer.txt');
  }

  /**
   * Load default user context
   */
  loadDefaultContext(): string {
    return this.loadPromptFile('default-context.txt');
  }

  /**
   * Load user-provided context from a file path
   */
  loadUserContext(contextPath: string): string {
    try {
      return readFileSync(contextPath, 'utf-8');
    } catch (error) {
      throw new Error(`Failed to load user context from ${contextPath}: ${error}`);
    }
  }

  /**
   * Build the complete prompt for Claude including all components
   */
  buildFullPrompt(
    cuCode: string,
    cuFilePath: string,
    userContext?: string,
    diagnosticContext?: string,
    customCommandsContext?: string
  ): string {
    const systemPrompt = this.loadSystemPrompt();
    const ncuAnalyzer = this.loadNCUAnalyzer();
    const codeOptimizer = this.loadCodeOptimizer();
    const context = userContext || this.loadDefaultContext();

    let fullPrompt = `${systemPrompt}

---

${ncuAnalyzer}

---

${codeOptimizer}

---

USER CONTEXT:
${context}`;

    // Add diagnostic context if provided
    if (diagnosticContext) {
      fullPrompt += `

---

${diagnosticContext}`;
    }

    // Add custom NCU commands if provided
    if (customCommandsContext) {
      fullPrompt += `

---

${customCommandsContext}`;
    }

    fullPrompt += `

---

CUDA KERNEL FILE: ${cuFilePath}

\`\`\`cuda
${cuCode}
\`\`\`

---

TASK (FOLLOW EVERY STEP - EXECUTION IS MANDATORY):

⚠️ YOU MUST EXECUTE NCU PROFILING - DO NOT SKIP THIS ⚠️

STEP 1: **COMPILE THE KERNEL**
   - Use nvcc to compile ${cuFilePath} into an executable
   - Command example: nvcc ${cuFilePath} -o kernel_executable
   - This is MANDATORY - you cannot profile without compilation

STEP 2: **EXECUTE NCU PROFILING**
   - You MUST run NCU profiling commands using the Bash tool
   - Use the ncu-llm wrapper script located at ./scripts/ncu-llm
   ${diagnosticContext ? '- OR use the specific NCU commands from the diagnostic configuration' : ''}
   ${customCommandsContext ? '- OR use the custom NCU commands provided by the user' : ''}
   - DO NOT skip profiling and analyze the code statically
   - Choose the appropriate mode: quick, bottleneck, or standard
   - Example: ./scripts/ncu-llm quick ./kernel_executable

STEP 3: **READ NCU OUTPUT**
   - Use the Read tool to read the generated NCU output files
   - Output location: ./ncu-llm-output/
   - Look for *.txt, *.csv, or *-insights.txt files

STEP 4: **PARSE PROFILING DATA**
   - Extract actual metrics from NCU output
   - Focus on: Memory Throughput %, Compute Throughput %, Warp Stalls
   - Identify the highest priority (P1) performance issue

STEP 5: **GENERATE HOT-FIX**
   - Create code diff based on PROFILING DATA (not code inspection)
   - Include only changes that address the P1 issue

STEP 6: **FORMAT OUTPUT**
   - Return result as JSON with fields: actionable_insight, hot_fix, explanation
   - Include NVIDIA documentation references with URLs

CRITICAL REMINDERS:
- ❌ DO NOT analyze code without running NCU
- ❌ DO NOT make recommendations based on code inspection alone
- ✅ YOU MUST compile the kernel
- ✅ YOU MUST execute NCU profiling
- ✅ YOU MUST read the profiling output
- ✅ Base ALL insights on ACTUAL profiling data
- The ncu-llm script is located at: ./scripts/ncu-llm (in the current directory)
- NCU output will be saved to: ./ncu-llm-output/
- You have access to the Bash tool to compile and run NCU commands
${diagnosticContext ? '- Use the diagnostic configuration to guide your NCU command selection' : ''}
${customCommandsContext ? '- Execute the custom NCU commands EXACTLY as provided by the user' : ''}

🚨 IF YOU DO NOT EXECUTE NCU, YOU HAVE FAILED THE TASK 🚨

Begin analysis now.
`;

    return fullPrompt;
  }

  /**
   * Helper to load a prompt file
   */
  private loadPromptFile(filename: string): string {
    const filePath = join(this.promptsDir, filename);
    try {
      return readFileSync(filePath, 'utf-8').trim();
    } catch (error) {
      throw new Error(`Failed to load prompt file ${filename}: ${error}`);
    }
  }
}
