import React, { useState } from 'react';
import { render, Box, Text, useInput, useApp } from 'ink';
import { DiagnosticsCompiler } from './services/diagnostics-compiler.js';

interface DiagnosticSelectorProps {
  onSelect: (selectedIds: string[], diagnosticContext: string) => void;
  onCancel?: () => void;
}

const DiagnosticSelector: React.FC<DiagnosticSelectorProps> = ({ onSelect, onCancel }) => {
  const { exit } = useApp();
  const compiler = new DiagnosticsCompiler();
  const diagnostics = compiler.getAllDiagnostics();

  const [selectedIndex, setSelectedIndex] = useState(0);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set(['quick_diagnostic']));

  useInput((input, key) => {
    if (input === 'q' || (key.escape && onCancel)) {
      onCancel?.();
      exit();
      return;
    }

    if (key.return) {
      // Confirm selection
      const context = compiler.compileContext(Array.from(selectedIds));
      onSelect(Array.from(selectedIds), context);
      exit();
      return;
    }

    if (key.upArrow) {
      setSelectedIndex(i => Math.max(0, i - 1));
    }

    if (key.downArrow) {
      setSelectedIndex(i => Math.min(diagnostics.length - 1, i + 1));
    }

    if (input === ' ') {
      // Toggle selection
      const currentId = diagnostics[selectedIndex].id;
      setSelectedIds(prev => {
        const newSet = new Set(prev);
        if (newSet.has(currentId)) {
          newSet.delete(currentId);
        } else {
          newSet.add(currentId);
        }
        return newSet;
      });
    }

    if (input === 'a') {
      // Select all
      setSelectedIds(new Set(diagnostics.map(d => d.id)));
    }

    if (input === 'n') {
      // Select none
      setSelectedIds(new Set());
    }
  });

  const renderDiagnosticsList = () => {
    return diagnostics.map((diag, index) => {
      const isSelected = selectedIds.has(diag.id);
      const isCursor = index === selectedIndex;
      const checkbox = isSelected ? '[✓]' : '[ ]';
      const cursor = isCursor ? '→ ' : '  ';

      const priorityColor = getPriorityColor(diag.priority);
      const overheadColor = getOverheadColor(diag.overhead);

      return (
        <Box key={diag.id} flexDirection="column" marginBottom={0}>
          <Text>
            {cursor}
            <Text color={isSelected ? 'green' : 'white'}>{checkbox}</Text>
            {' '}
            <Text bold color={isCursor ? 'cyan' : 'white'}>{diag.name}</Text>
            {' '}
            <Text color={priorityColor}>[{diag.priority.toUpperCase()}]</Text>
            {' '}
            <Text color={overheadColor}>({diag.overhead} overhead)</Text>
          </Text>
          {isCursor && (
            <Text color="gray">      {diag.description.slice(0, 70)}...</Text>
          )}
        </Box>
      );
    });
  };

  return (
    <Box flexDirection="column" padding={1}>
      {/* Header */}
      <Box borderStyle="round" borderColor="cyan" flexDirection="column" padding={1} marginBottom={1}>
        <Text bold color="cyan">📊 NCU Diagnostic Configuration</Text>
        <Text color="gray">
          Select diagnostics to guide Claude's NCU profiling strategy
        </Text>
      </Box>

      {/* Instructions */}
      <Box borderStyle="single" borderColor="yellow" flexDirection="column" padding={1} marginBottom={1}>
        <Text bold color="yellow">Controls:</Text>
        <Text>
          <Text color="cyan">↑/↓</Text> Navigate  {' '}
          <Text color="cyan">Space</Text> Toggle  {' '}
          <Text color="cyan">A</Text> Select All  {' '}
          <Text color="cyan">N</Text> Select None  {' '}
          <Text color="cyan">Enter</Text> Confirm  {' '}
          <Text color="cyan">Q</Text> Cancel
        </Text>
      </Box>

      {/* Diagnostics List */}
      <Box borderStyle="round" borderColor="magenta" flexDirection="column" padding={1} marginBottom={1}>
        <Text bold color="magenta">
          Available Diagnostics ({selectedIds.size} selected)
        </Text>
        <Text> </Text>
        {renderDiagnosticsList()}
      </Box>

      {/* Footer */}
      <Box borderStyle="single" borderColor="green" padding={1}>
        <Text color="green">
          Press <Text bold>Enter</Text> to continue with {selectedIds.size} diagnostic{selectedIds.size !== 1 ? 's' : ''} selected
        </Text>
      </Box>
    </Box>
  );
};

function getPriorityColor(priority: string): string {
  switch (priority.toLowerCase()) {
    case 'critical':
      return 'red';
    case 'high':
      return 'yellow';
    case 'medium':
      return 'cyan';
    case 'low':
      return 'gray';
    default:
      return 'white';
  }
}

function getOverheadColor(overhead: string): string {
  switch (overhead.toLowerCase()) {
    case 'high':
      return 'red';
    case 'medium':
      return 'yellow';
    case 'low':
      return 'green';
    case 'none':
      return 'gray';
    default:
      return 'white';
  }
}

/**
 * Launch the diagnostic selector TUI
 */
export function launchDiagnosticSelector(): Promise<{
  selectedIds: string[];
  context: string;
} | null> {
  return new Promise((resolve) => {
    const { waitUntilExit } = render(
      <DiagnosticSelector
        onSelect={(selectedIds, context) => {
          resolve({ selectedIds, context });
        }}
        onCancel={() => {
          resolve(null);
        }}
      />
    );

    waitUntilExit();
  });
}
