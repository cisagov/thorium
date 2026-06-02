export function removeLine(text: string, prefix: string): string {
  return text
    .split('\n')
    .filter((line) => !line.trimStart().startsWith(prefix))
    .join('\n');
}

export function removeBlock(text: string, key: string): string {
  const lines = text.split('\n');
  const result: string[] = [];
  let skipping = false;
  for (const line of lines) {
    if (line.startsWith(`${key}:`)) {
      skipping = true;
      continue;
    }
    if (skipping && (line.startsWith('    ') || line.startsWith('\t') || line.trim() === '')) {
      continue;
    }
    skipping = false;
    result.push(line);
  }
  return result.join('\n');
}

export function replaceLine(text: string, prefix: string, replacement: string): string {
  return text
    .split('\n')
    .map((line) => (line.trimStart().startsWith(prefix) ? replacement : line))
    .join('\n');
}
