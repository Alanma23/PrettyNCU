#!/usr/bin/env node

import { Command } from 'commander';
import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';
import { launchTUI } from './tui.js';
import { launchDiagnosticSelector } from './diagnostic-selector.js';
import chalk from 'chalk';

const program = new Command();

program
  .name('ncu-claude')
  .description('Intelligent NCU profiling with Claude AI')
  .version('1.0.0');

program
  .command('analyze')
  .description('Analyze a CUDA kernel file with NCU profiling')
  .argument('<file>', 'CUDA kernel file (.cu) to analyze')
  .option('-c, --context <file>', 'Custom context file with optimization preferences')
  .option('-i, --context-inline <text>', 'Inline context text')
  .option('--configure-diagnostics', 'Launch interactive diagnostic configuration selector')
  .option('--ncu-command <command>', 'Custom NCU command to execute (e.g., "ncu --metrics sm__warps_active.avg -o output")')
  .option('--ncu-commands-file <file>', 'File containing custom NCU commands (one per line)')
  .option('--interactive', 'Interactive mode - prompts for context input')
  .action(async (file: string, options: any) => {
    // Validate file exists
    const cuFilePath = resolve(file);
    if (!existsSync(cuFilePath)) {
      console.error(chalk.red(`Error: File not found: ${cuFilePath}`));
      process.exit(1);
    }

    // Validate file extension
    if (!cuFilePath.endsWith('.cu')) {
      console.warn(chalk.yellow(`Warning: File does not have .cu extension: ${cuFilePath}`));
    }

    // Validate context file if provided
    if (options.context) {
      const contextPath = resolve(options.context);
      if (!existsSync(contextPath)) {
        console.error(chalk.red(`Error: Context file not found: ${contextPath}`));
        process.exit(1);
      }
      options.context = contextPath;
    }

    // Handle diagnostic configuration
    let diagnosticContext: string | undefined;
    if (options.configureDiagnostics) {
      console.log(chalk.cyan('Launching diagnostic configuration selector...\n'));
      const result = await launchDiagnosticSelector();

      if (!result) {
        console.log(chalk.yellow('Diagnostic configuration canceled'));
        process.exit(0);
      }

      diagnosticContext = result.context;
      console.log(chalk.green(`\n✓ Selected ${result.selectedIds.length} diagnostic(s)\n`));
    }

    // Handle custom NCU commands
    let customNcuCommands: string[] = [];

    if (options.ncuCommand) {
      customNcuCommands.push(options.ncuCommand);
      console.log(chalk.cyan(`Custom NCU command: ${options.ncuCommand}\n`));
    }

    if (options.ncuCommandsFile) {
      const commandsPath = resolve(options.ncuCommandsFile);
      if (!existsSync(commandsPath)) {
        console.error(chalk.red(`Error: NCU commands file not found: ${commandsPath}`));
        process.exit(1);
      }

      try {
        const fileContent = readFileSync(commandsPath, 'utf-8');
        const commands = fileContent
          .split('\n')
          .map(line => line.trim())
          .filter(line => line.length > 0 && !line.startsWith('#')); // Filter empty lines and comments

        customNcuCommands.push(...commands);
        console.log(chalk.cyan(`Loaded ${commands.length} custom NCU command(s) from file\n`));
      } catch (error) {
        console.error(chalk.red(`Error reading NCU commands file: ${error}`));
        process.exit(1);
      }
    }

    // Build custom commands context if provided
    let customCommandsContext: string | undefined;
    if (customNcuCommands.length > 0) {
      customCommandsContext = buildCustomCommandsContext(customNcuCommands);
    }

    // Launch TUI
    try {
      await launchTUI({
        cuFilePath,
        contextPath: options.context,
        contextInline: options.contextInline,
        diagnosticContext,
        customCommandsContext,
        interactive: options.interactive
      });
    } catch (error) {
      console.error(chalk.red(`Error: ${error}`));
      process.exit(1);
    }
  });

/**
 * Build context for custom NCU commands
 */
function buildCustomCommandsContext(commands: string[]): string {
  let context = 'CUSTOM NCU COMMANDS\n';
  context += '='.repeat(80) + '\n\n';
  context += 'The user has provided the following custom NCU commands to execute:\n\n';

  commands.forEach((cmd, index) => {
    context += `${index + 1}. ${cmd}\n`;
  });

  context += '\nIMPORTANT INSTRUCTIONS:\n';
  context += '- Execute these EXACT commands as provided by the user\n';
  context += '- Do NOT modify the command flags or parameters\n';
  context += '- If a command specifies an output file with -o, use that path\n';
  context += '- After executing, read and parse the output files\n';
  context += '- Base your analysis on the metrics from these custom commands\n';
  context += '- If the commands fail, report the error to the user\n\n';

  return context;
}

// Add diagnostics configuration command
program
  .command('configure')
  .description('Configure NCU diagnostics interactively')
  .action(async () => {
    console.log(chalk.cyan('Launching diagnostic configuration selector...\n'));
    const result = await launchDiagnosticSelector();

    if (!result) {
      console.log(chalk.yellow('Configuration canceled'));
      process.exit(0);
    }

    console.log(chalk.green(`\n✓ Selected ${result.selectedIds.length} diagnostic(s):`));
    console.log(result.selectedIds.map(id => '  - ' + id).join('\n'));
    console.log(chalk.dim('\nUse --configure-diagnostics flag when running analyze to select diagnostics.'));
  });

// Show help if no command provided
if (process.argv.length === 2) {
  program.help();
}

program.parse();
