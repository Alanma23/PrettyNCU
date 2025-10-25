import { query } from '@anthropic-ai/claude-agent-sdk';
import type { SDKMessage } from '@anthropic-ai/claude-agent-sdk';
import { PromptLoader } from './prompt-loader.js';

export interface NCUAnalysisResult {
  actionable_insight: string;
  hot_fix: string;
  explanation: string;
  ncu_metadata?: {
    commands_executed: string[];
    output_files: string[];
    raw_data_snippet?: string;
  };
}

export interface AnalysisProgress {
  type: 'compilation' | 'claude_thinking' | 'ncu_execution' | 'parsing' | 'complete' | 'error' | 'tool_use';
  message: string;
  data?: any;
  toolDetails?: {
    toolName: string;
    command?: string;
    description?: string;
    input?: any;
  };
}

export class ClaudeAgent {
  private promptLoader: PromptLoader;

  constructor() {
    this.promptLoader = new PromptLoader();
  }

  /**
   * Analyze a CUDA kernel file with Claude and NCU profiling
   * Returns an async generator that yields progress updates
   */
  async *analyzeKernel(
    cuFilePath: string,
    cuCode: string,
    userContext?: string,
    diagnosticContext?: string,
    customCommandsContext?: string,
    onProgress?: (progress: AnalysisProgress) => void
  ): AsyncGenerator<AnalysisProgress, NCUAnalysisResult | null, unknown> {
    try {
      // Build the full prompt
      const prompt = this.promptLoader.buildFullPrompt(
        cuCode,
        cuFilePath,
        userContext,
        diagnosticContext,
        customCommandsContext
      );

      // Emit initial progress
      yield {
        type: 'claude_thinking',
        message: 'Starting Claude analysis...'
      };

      // Initialize the query with Claude Agent SDK
      const result = query({
        prompt: prompt,
        options: {
          model: 'claude-sonnet-4-5-20250929',
          permissionMode: 'bypassPermissions', // Auto-execute NCU commands
          cwd: process.cwd(),
          maxThinkingTokens: 10000,
        }
      });

      let finalResult: NCUAnalysisResult | null = null;
      let assistantMessages: string[] = [];
      const ncuCommands: string[] = [];
      const outputFiles: string[] = [];

      // Stream messages from Claude
      for await (const message of result) {
        const sdkMessage = message as SDKMessage;

        if (sdkMessage.type === 'assistant') {
          // Extract text content from assistant message
          const content = sdkMessage.message.content;
          let textContent = '';

          if (Array.isArray(content)) {
            for (const block of content) {
              if (block.type === 'text') {
                textContent += block.text;
              } else if (block.type === 'tool_use') {
                // Claude is using a tool (e.g., Bash to run NCU)
                const toolInput = block.input || {};
                const toolName = block.name;

                // Extract specific details based on tool type
                let command = '';
                let description = '';

                if (toolName === 'Bash') {
                  command = toolInput.command || '';
                  description = toolInput.description || '';

                  // Track NCU commands
                  if (command.includes('ncu')) {
                    ncuCommands.push(command);

                    // Extract output file path from -o flag
                    const outputMatch = command.match(/-o\s+(\S+)/);
                    if (outputMatch) {
                      const outputPath = outputMatch[1];
                      // NCU creates multiple files with extensions
                      outputFiles.push(`${outputPath}.txt`);
                      outputFiles.push(`${outputPath}.csv`);
                      outputFiles.push(`${outputPath}.ncu-rep`);
                    }
                  }
                } else if (toolName === 'Read') {
                  command = `Reading: ${toolInput.file_path || ''}`;
                  description = 'Reading file';
                } else if (toolName === 'Write') {
                  command = `Writing to: ${toolInput.file_path || ''}`;
                  description = 'Writing file';
                } else if (toolName === 'Edit') {
                  command = `Editing: ${toolInput.file_path || ''}`;
                  description = 'Editing file';
                } else if (toolName === 'Grep') {
                  command = `Searching for: ${toolInput.pattern || ''}`;
                  description = toolInput.description || 'Searching code';
                } else {
                  command = JSON.stringify(toolInput).slice(0, 100);
                  description = `Executing ${toolName}`;
                }

                yield {
                  type: 'tool_use',
                  message: `🔧 ${toolName}: ${description || command}`,
                  toolDetails: {
                    toolName,
                    command,
                    description,
                    input: toolInput
                  }
                };
              }
            }
          }

          if (textContent) {
            assistantMessages.push(textContent);

            // Try to detect what phase we're in from the message
            const lowerText = textContent.toLowerCase();
            if (lowerText.includes('compiling') || lowerText.includes('nvcc')) {
              yield {
                type: 'compilation',
                message: 'Compiling CUDA kernel...'
              };
            } else if (lowerText.includes('ncu') || lowerText.includes('profiling')) {
              yield {
                type: 'ncu_execution',
                message: 'Running NCU profiling...'
              };
            } else if (lowerText.includes('parsing') || lowerText.includes('analyzing')) {
              yield {
                type: 'parsing',
                message: 'Analyzing NCU output...'
              };
            } else {
              yield {
                type: 'claude_thinking',
                message: textContent.slice(0, 100) + (textContent.length > 100 ? '...' : '')
              };
            }
          }
        } else if (sdkMessage.type === 'result') {
          // Final result from Claude
          yield {
            type: 'parsing',
            message: 'Extracting final results...'
          };

          // Try to extract JSON from the assistant messages
          const fullResponse = assistantMessages.join('\n');
          finalResult = this.extractJSONResult(fullResponse);

          // Add NCU metadata to the result
          if (finalResult && (ncuCommands.length > 0 || outputFiles.length > 0)) {
            finalResult.ncu_metadata = {
              commands_executed: ncuCommands,
              output_files: outputFiles.filter((file, index, self) => self.indexOf(file) === index), // Remove duplicates
              raw_data_snippet: this.extractNCUDataSnippet(outputFiles)
            };
          }

          if (finalResult) {
            yield {
              type: 'complete',
              message: 'Analysis complete!',
              data: finalResult
            };
          } else {
            yield {
              type: 'error',
              message: 'Failed to extract JSON result from Claude response'
            };
          }
        }
      }

      return finalResult;
    } catch (error) {
      yield {
        type: 'error',
        message: `Analysis failed: ${error}`
      };
      return null;
    }
  }

  /**
   * Extract the JSON result from Claude's response
   */
  private extractJSONResult(response: string): NCUAnalysisResult | null {
    try {
      // Try to find JSON object in the response
      // Look for ```json blocks first
      const jsonBlockMatch = response.match(/```json\s*(\{[\s\S]*?\})\s*```/);
      if (jsonBlockMatch) {
        return JSON.parse(jsonBlockMatch[1]);
      }

      // Try to find raw JSON object
      const jsonMatch = response.match(/\{[\s\S]*"actionable_insight"[\s\S]*"hot_fix"[\s\S]*"explanation"[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }

      // If no JSON found, try to construct it from the response
      // This is a fallback for cases where Claude doesn't format it as JSON
      const insightMatch = response.match(/actionable_insight["\s:]+(.+?)(?=\n|hot_fix|$)/i);
      const hotFixMatch = response.match(/hot_fix["\s:]+(.+?)(?=\n|explanation|$)/i);
      const explanationMatch = response.match(/explanation["\s:]+(.+?)$/is);

      if (insightMatch && hotFixMatch && explanationMatch) {
        return {
          actionable_insight: insightMatch[1].trim(),
          hot_fix: hotFixMatch[1].trim(),
          explanation: explanationMatch[1].trim()
        };
      }

      return null;
    } catch (error) {
      console.error('Failed to parse JSON result:', error);
      return null;
    }
  }

  /**
   * Extract a snippet of raw NCU data from output files
   */
  private extractNCUDataSnippet(outputFiles: string[]): string | undefined {
    try {
      const fs = require('fs');

      // Try to read the first .txt file
      const txtFile = outputFiles.find(f => f.endsWith('.txt'));
      if (txtFile && fs.existsSync(txtFile)) {
        const content = fs.readFileSync(txtFile, 'utf-8');
        // Return first 500 characters
        return content.slice(0, 500) + (content.length > 500 ? '\n...(truncated)' : '');
      }

      // If no .txt file, try .csv
      const csvFile = outputFiles.find(f => f.endsWith('.csv'));
      if (csvFile && fs.existsSync(csvFile)) {
        const content = fs.readFileSync(csvFile, 'utf-8');
        const lines = content.split('\n').slice(0, 10); // First 10 lines
        return lines.join('\n') + '\n...(truncated)';
      }

      return undefined;
    } catch (error) {
      return `Error reading NCU output: ${error}`;
    }
  }
}
