import React, { useState, useEffect } from 'react';
import { render, Box, Text, Newline } from 'ink';
import TextInput from 'ink-text-input';
import chalk from 'chalk';
import { ClaudeAgent, type NCUAnalysisResult, type AnalysisProgress } from './services/claude-agent.js';
import { formatHotFix, extractCodeFromDiffString } from './utils/side-by-side-diff.js';
import { readFileSync } from 'fs';

interface AnalyzerAppProps {
  cuFilePath: string;
  contextPath?: string;
  contextInline?: string;
  diagnosticContext?: string;
  customCommandsContext?: string;
  interactive?: boolean;
}

const AnalyzerApp: React.FC<AnalyzerAppProps> = ({
  cuFilePath,
  contextPath,
  contextInline,
  diagnosticContext,
  customCommandsContext,
  interactive = false
}) => {
  const [stage, setStage] = useState<'input' | 'analyzing' | 'complete'>('input');
  const [userContext, setUserContext] = useState<string>('');
  const [progress, setProgress] = useState<AnalysisProgress[]>([]);
  const [result, setResult] = useState<NCUAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load context
  useEffect(() => {
    if (contextInline) {
      setUserContext(contextInline);
      if (!interactive) {
        setStage('analyzing');
      }
    } else if (contextPath) {
      try {
        const ctx = readFileSync(contextPath, 'utf-8');
        setUserContext(ctx);
        if (!interactive) {
          setStage('analyzing');
        }
      } catch (err) {
        setError(`Failed to load context from ${contextPath}: ${err}`);
      }
    } else if (!interactive) {
      // Use default context, start immediately
      setStage('analyzing');
    }
  }, [contextPath, contextInline, interactive]);

  // Run analysis
  useEffect(() => {
    if (stage === 'analyzing') {
      runAnalysis();
    }
  }, [stage]);

  const runAnalysis = async () => {
    try {
      const cuCode = readFileSync(cuFilePath, 'utf-8');
      const agent = new ClaudeAgent();

      const progressUpdates: AnalysisProgress[] = [];

      for await (const update of agent.analyzeKernel(
        cuFilePath,
        cuCode,
        userContext || undefined,
        diagnosticContext || undefined,
        customCommandsContext || undefined
      )) {
        progressUpdates.push(update);
        setProgress([...progressUpdates]);

        if (update.type === 'complete' && update.data) {
          setResult(update.data as NCUAnalysisResult);
          setStage('complete');
        } else if (update.type === 'error') {
          setError(update.message);
          setStage('complete');
        }
      }
    } catch (err) {
      setError(`Analysis failed: ${err}`);
      setStage('complete');
    }
  };

  const handleContextSubmit = (value: string) => {
    setUserContext(value);
    setStage('analyzing');
  };

  // Render input stage
  if (stage === 'input') {
    return (
      <Box flexDirection="column" padding={1}>
        <Box borderStyle="round" borderColor="cyan" flexDirection="column" padding={1}>
          <Text bold color="cyan">NCU Claude Analyzer</Text>
          <Newline />
          <Text>File: <Text color="green">{cuFilePath}</Text></Text>
          <Newline />
          <Text>Additional context (optional, press Enter to use defaults):</Text>
          <Box marginTop={1}>
            <TextInput
              value={userContext}
              onChange={setUserContext}
              onSubmit={handleContextSubmit}
              placeholder="e.g., Focus on L1 cache, Target A100 GPU..."
            />
          </Box>
        </Box>
      </Box>
    );
  }

  // Render analyzing stage
  if (stage === 'analyzing') {
    return (
      <Box flexDirection="column" padding={1}>
        <Box borderStyle="round" borderColor="cyan" flexDirection="column" padding={1}>
          <Text bold color="cyan">🔍 Analyzing {cuFilePath}...</Text>
          <Newline />
          {progress.map((p, i) => {
            // Special formatting for tool_use type with detailed info
            if (p.type === 'tool_use' && p.toolDetails) {
              const { toolName, command, description } = p.toolDetails;
              return (
                <Box key={i} flexDirection="column" marginBottom={0}>
                  <Text color="cyan">
                    🔧 <Text bold>{toolName}</Text>
                  </Text>
                  {command && (
                    <Text color="gray">   └─ {truncateCommand(command, 70)}</Text>
                  )}
                </Box>
              );
            }

            // Default formatting
            return (
              <Box key={i} marginBottom={0}>
                <Text color={p.type === 'error' ? 'red' : 'white'}>
                  {getProgressIcon(p.type)} {p.message}
                </Text>
              </Box>
            );
          })}
        </Box>
      </Box>
    );
  }

  // Render complete stage
  if (stage === 'complete') {
    if (error) {
      return (
        <Box flexDirection="column" padding={1}>
          <Box borderStyle="round" borderColor="red" flexDirection="column" padding={1}>
            <Text bold color="red">❌ Error</Text>
            <Newline />
            <Text>{error}</Text>
          </Box>
        </Box>
      );
    }

    if (!result) {
      return (
        <Box flexDirection="column" padding={1}>
          <Box borderStyle="round" borderColor="yellow" flexDirection="column" padding={1}>
            <Text bold color="yellow">⚠️  No results</Text>
            <Newline />
            <Text>Analysis completed but no structured result was returned.</Text>
          </Box>
        </Box>
      );
    }

    return (
      <Box flexDirection="column" padding={1}>
        {/* Header */}
        <Box borderStyle="round" borderColor="green" flexDirection="column" padding={1} marginBottom={1}>
          <Text bold color="green">✅ Analysis Complete</Text>
        </Box>

        {/* Actionable Insight */}
        <Box borderStyle="round" borderColor="yellow" flexDirection="column" padding={1} marginBottom={1}>
          <Text bold color="yellow">⚡ P1 Actionable Insight:</Text>
          <Newline />
          <Text>{result.actionable_insight}</Text>
        </Box>

        {/* Hot Fix - Side-by-side diff */}
        <Box borderStyle="round" borderColor="magenta" flexDirection="column" padding={1} marginBottom={1}>
          <Text bold color="magenta">🔧 Hot Fix (Side-by-Side Diff):</Text>
          <Newline />
          <Text>{renderSideBySideDiff(result.hot_fix, cuFilePath)}</Text>
        </Box>

        {/* Explanation */}
        <Box borderStyle="round" borderColor="blue" flexDirection="column" padding={1} marginBottom={1}>
          <Text bold color="blue">📖 Explanation:</Text>
          <Newline />
          <Text>{result.explanation}</Text>
        </Box>

        {/* NCU Metadata */}
        {result.ncu_metadata && (
          <Box borderStyle="round" borderColor="cyan" flexDirection="column" padding={1}>
            <Text bold color="cyan">📊 NCU Profiling Information:</Text>
            <Newline />

            {result.ncu_metadata.commands_executed.length > 0 && (
              <>
                <Text bold color="green">Commands Executed:</Text>
                {result.ncu_metadata.commands_executed.map((cmd, idx) => (
                  <Text key={idx} color="gray">  {idx + 1}. {cmd}</Text>
                ))}
                <Newline />
              </>
            )}

            {result.ncu_metadata.output_files.length > 0 && (
              <>
                <Text bold color="green">Output Files:</Text>
                {result.ncu_metadata.output_files.map((file, idx) => (
                  <Text key={idx} color="gray">  • {file}</Text>
                ))}
                <Newline />
              </>
            )}

            {result.ncu_metadata.raw_data_snippet && (
              <>
                <Text bold color="green">Raw NCU Data (snippet):</Text>
                <Text color="gray">{result.ncu_metadata.raw_data_snippet}</Text>
              </>
            )}
          </Box>
        )}
      </Box>
    );
  }

  return null;
};

function getProgressIcon(type: string): string {
  switch (type) {
    case 'compilation':
      return '🔨';
    case 'claude_thinking':
      return '🤔';
    case 'ncu_execution':
      return '📊';
    case 'tool_use':
      return '🔧';
    case 'parsing':
      return '📝';
    case 'complete':
      return '✅';
    case 'error':
      return '❌';
    default:
      return '⏳';
  }
}

function truncateCommand(cmd: string, maxLen: number): string {
  if (cmd.length <= maxLen) return cmd;
  return cmd.substring(0, maxLen - 3) + '...';
}

function renderSideBySideDiff(hotFix: string, cuFilePath: string): string {
  // Try to extract original code from the cuFilePath if possible
  try {
    const originalCode = readFileSync(cuFilePath, 'utf-8');
    const { original, modified } = extractCodeFromDiffString(hotFix);

    if (original && modified) {
      return formatHotFix(hotFix, originalCode);
    }
  } catch (err) {
    // Fall through to default rendering
  }

  // Default: just format the hot fix as-is
  return formatHotFix(hotFix);
}

/**
 * Launch the TUI
 */
export function launchTUI(options: AnalyzerAppProps) {
  const { waitUntilExit } = render(<AnalyzerApp {...options} />);
  return waitUntilExit();
}
