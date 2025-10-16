import { mkdtemp, rm, readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { tmpdir } from 'node:os';

import { afterEach, describe, expect, test } from 'vitest';

import { generateNotebook } from '../../src/generateNotebook.js';
import fullConfig from '../../examples/configs/full.json' assert { type: 'json' };
import loraConfig from '../../examples/configs/lora.json' assert { type: 'json' };
import qloraConfig from '../../examples/configs/qlora.json' assert { type: 'json' };

const tempRoots: string[] = [];

afterEach(async () => {
  while (tempRoots.length > 0) {
    const dir = tempRoots.pop();
    if (dir) {
      await rm(dir, { recursive: true, force: true });
    }
  }
});

describe('generateNotebook', () => {
  test('creates notebook and zip bundle for full config', async () => {
    const outputDir = await prepareTempDir('codex-full-');
    const result = await generateNotebook(fullConfig, { outputDir });

    await expectPathExists(result.notebookPath);
    await expectPathExists(result.configPath);
    await expectPathExists(result.readmePath);
    expect(result.zipPath).toBeTruthy();
    if (result.zipPath) {
      await expectPathExists(result.zipPath);
    }

    const notebookRaw = await readFile(result.notebookPath, 'utf8');
    const notebook = JSON.parse(notebookRaw) as { cells: Array<Record<string, unknown>> };
    const tags = notebook.cells.flatMap((cell) => {
      const metadata = cell.metadata as { tags?: unknown } | undefined;
      return Array.isArray(metadata?.tags) ? (metadata?.tags as string[]) : [];
    });
    expect(tags).toContain('overview');
    expect(tags).toContain('setup');
    expect(tags).toContain('training');

    const configCell = notebook.cells.find((cell) => {
      const metadata = cell.metadata as { tags?: unknown } | undefined;
      return Array.isArray(metadata?.tags) && (metadata.tags as unknown[]).includes('config');
    });
    expect(configCell).toBeDefined();
    const source = (configCell?.source ?? []) as unknown[];
    expect(Array.isArray(source)).toBe(true);
    expect(source.some((line) => typeof line === 'string' && line.includes('CONFIG_GENERATED_AT'))).toBe(true);
  });

  test('supports disabling zip bundle', async () => {
    const outputDir = await prepareTempDir('codex-lora-');
    const result = await generateNotebook(loraConfig, {
      outputDir,
      createZip: false
    });

    await expectPathExists(result.notebookPath);
    expect(result.zipPath).toBeUndefined();

    const readme = await readFile(result.readmePath, 'utf8');
    expect(readme).toContain('Fine-Tuning Notebook Bundle');
    expect(readme).toContain('Provider');
  });

  test('requires tune-type specific sections', async () => {
    const outputDir = await prepareTempDir('codex-qlora-');
    const invalid = JSON.parse(JSON.stringify(qloraConfig));
    delete invalid.qlora;

    await expect(
      generateNotebook(invalid, {
        outputDir
      })
    ).rejects.toThrow(/Config validation failed/);
  });
});

async function prepareTempDir(prefix: string): Promise<string> {
  const dir = await mkdtemp(path.join(tmpdir(), prefix));
  tempRoots.push(dir);
  return dir;
}

async function expectPathExists(filePath: string): Promise<void> {
  const stats = await stat(filePath);
  expect(stats.isFile()).toBe(true);
}
