import { diffLines } from 'diff';
import chalk from 'chalk';

export interface SideBySideDiffOptions {
  leftTitle?: string;
  rightTitle?: string;
  maxWidth?: number;
  contextLines?: number;
}

/**
 * Generate a side-by-side diff view for terminal display
 */
export function generateSideBySideDiff(
  original: string,
  modified: string,
  options: SideBySideDiffOptions = {}
): string {
  const {
    leftTitle = 'Original',
    rightTitle = 'Optimized',
    maxWidth = 80,
    contextLines = 3
  } = options;

  const originalLines = original.split('\n');
  const modifiedLines = modified.split('\n');
  const diff = diffLines(original, modified);

  const output: string[] = [];
  const columnWidth = Math.floor((maxWidth - 5) / 2); // 5 for separator and padding

  // Header
  const headerSeparator = '═'.repeat(maxWidth);
  output.push(chalk.bold(headerSeparator));
  output.push(
    chalk.bold.cyan(leftTitle.padEnd(columnWidth)) +
    chalk.gray(' │ ') +
    chalk.bold.green(rightTitle.padEnd(columnWidth))
  );
  output.push(chalk.bold(headerSeparator));

  let leftLineNum = 1;
  let rightLineNum = 1;

  for (const part of diff) {
    const lines = part.value.split('\n').filter(line => line.trim().length > 0 || part.added || part.removed);

    if (part.removed) {
      // Lines removed from original (shown on left only)
      for (const line of lines) {
        const leftText = truncate(line, columnWidth - 4);
        const leftFormatted = chalk.red(`${String(leftLineNum).padStart(3)} ${leftText}`);
        const rightFormatted = ' '.repeat(columnWidth);

        output.push(leftFormatted.padEnd(columnWidth + 10) + chalk.gray(' │ ') + rightFormatted);
        leftLineNum++;
      }
    } else if (part.added) {
      // Lines added to modified (shown on right only)
      for (const line of lines) {
        const rightText = truncate(line, columnWidth - 4);
        const leftFormatted = ' '.repeat(columnWidth);
        const rightFormatted = chalk.green(`${String(rightLineNum).padStart(3)} ${rightText}`);

        output.push(leftFormatted + chalk.gray(' │ ') + rightFormatted);
        rightLineNum++;
      }
    } else {
      // Unchanged lines (shown on both sides)
      for (const line of lines) {
        if (line.trim().length === 0) continue;

        const text = truncate(line, columnWidth - 4);
        const leftFormatted = chalk.gray(`${String(leftLineNum).padStart(3)} ${text}`);
        const rightFormatted = chalk.gray(`${String(rightLineNum).padStart(3)} ${text}`);

        output.push(leftFormatted.padEnd(columnWidth + 10) + chalk.gray(' │ ') + rightFormatted);
        leftLineNum++;
        rightLineNum++;
      }
    }
  }

  output.push(chalk.bold(headerSeparator));

  return output.join('\n');
}

/**
 * Generate a compact unified diff with better formatting
 */
export function generateUnifiedDiff(original: string, modified: string): string {
  const diff = diffLines(original, modified);
  const output: string[] = [];

  output.push(chalk.bold.cyan('─'.repeat(80)));
  output.push(chalk.bold('Code Changes:'));
  output.push(chalk.bold.cyan('─'.repeat(80)));

  for (const part of diff) {
    const lines = part.value.split('\n').filter(line => line.length > 0);

    for (const line of lines) {
      if (part.added) {
        output.push(chalk.green('+ ' + line));
      } else if (part.removed) {
        output.push(chalk.red('- ' + line));
      } else {
        output.push(chalk.gray('  ' + line));
      }
    }
  }

  output.push(chalk.bold.cyan('─'.repeat(80)));

  return output.join('\n');
}

/**
 * Extract original and modified code from a diff string
 */
export function extractCodeFromDiffString(diffText: string): {
  original: string;
  modified: string;
} {
  const lines = diffText.split('\n');
  const originalLines: string[] = [];
  const modifiedLines: string[] = [];

  for (const line of lines) {
    const trimmed = line.trim();

    // Skip diff headers and separators
    if (trimmed.startsWith('@@') || trimmed.startsWith('---') || trimmed.startsWith('+++')) {
      continue;
    }

    if (trimmed.startsWith('-')) {
      // Removed line
      originalLines.push(trimmed.substring(1).trim());
    } else if (trimmed.startsWith('+')) {
      // Added line
      modifiedLines.push(trimmed.substring(1).trim());
    } else if (trimmed.length > 0) {
      // Context line (unchanged)
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
 * Truncate a string to fit within a width, adding ellipsis if needed
 */
function truncate(str: string, maxLength: number): string {
  if (str.length <= maxLength) {
    return str.padEnd(maxLength);
  }
  return str.substring(0, maxLength - 3) + '...';
}

/**
 * Format a hot-fix for display with enhanced styling
 */
export function formatHotFix(hotFix: string, original: string = ''): string {
  const output: string[] = [];

  output.push('\n' + chalk.bold.magenta('🔧 Hot Fix:'));
  output.push(chalk.magenta('─'.repeat(80)));

  // Try to extract original and modified from the hot fix
  if (original) {
    const { original: origCode, modified: modCode } = extractCodeFromDiffString(hotFix);
    if (origCode && modCode) {
      output.push(generateSideBySideDiff(origCode, modCode, {
        leftTitle: 'Current Code',
        rightTitle: 'Optimized Code'
      }));
      return output.join('\n');
    }
  }

  // Fallback to unified diff display
  const lines = hotFix.split('\n');
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('@@')) {
      output.push(chalk.cyan(line));
    } else if (trimmed.startsWith('+')) {
      output.push(chalk.green(line));
    } else if (trimmed.startsWith('-')) {
      output.push(chalk.red(line));
    } else {
      output.push(chalk.gray(line));
    }
  }

  output.push(chalk.magenta('─'.repeat(80)));

  return output.join('\n');
}
