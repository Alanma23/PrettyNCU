import { diffLines } from 'diff';
import chalk from 'chalk';

/**
 * Format a unified diff for terminal display with colors
 */
export function formatDiff(oldCode: string, newCode: string): string {
  const diff = diffLines(oldCode, newCode);

  const formattedLines: string[] = [];

  for (const part of diff) {
    const lines = part.value.split('\n').filter(line => line.length > 0);

    for (const line of lines) {
      if (part.added) {
        formattedLines.push(chalk.green('+ ' + line));
      } else if (part.removed) {
        formattedLines.push(chalk.red('- ' + line));
      } else {
        formattedLines.push(chalk.gray('  ' + line));
      }
    }
  }

  return formattedLines.join('\n');
}

/**
 * Extract code blocks from a diff text
 */
export function extractCodeFromDiff(diffText: string): { original: string; modified: string } {
  const lines = diffText.split('\n');
  const originalLines: string[] = [];
  const modifiedLines: string[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('-')) {
      originalLines.push(trimmed.substring(1).trim());
    } else if (trimmed.startsWith('+')) {
      modifiedLines.push(trimmed.substring(1).trim());
    } else if (!trimmed.startsWith('@@') && trimmed.length > 0) {
      // Context line
      originalLines.push(trimmed);
      modifiedLines.push(trimmed);
    }
  }

  return {
    original: originalLines.join('\n'),
    modified: modifiedLines.join('\n')
  };
}

/**
 * Pretty print a hot-fix diff
 */
export function displayHotFix(hotFix: string): string {
  const lines = hotFix.split('\n');
  const output: string[] = [];

  output.push(chalk.bold('\n📝 Hot Fix:\n'));

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('@@')) {
      // File header
      output.push(chalk.cyan(line));
    } else if (trimmed.startsWith('+')) {
      // Added line
      output.push(chalk.green(line));
    } else if (trimmed.startsWith('-')) {
      // Removed line
      output.push(chalk.red(line));
    } else {
      // Context line
      output.push(chalk.gray(line));
    }
  }

  return output.join('\n');
}
