import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

import { renderBundle, type JsonValue } from './core/renderBundle.js';

export interface GenerateNotebookOptions {
  outputDir: string;
  createZip?: boolean;
  zipFileName?: string;
  timestamp?: Date;
}

export interface GenerationResult {
  notebookPath: string;
  configPath: string;
  readmePath: string;
  zipPath?: string;
}

export async function generateNotebook(
  config: JsonValue,
  options: GenerateNotebookOptions
): Promise<GenerationResult> {
  const {
    outputDir,
    createZip = true,
    zipFileName = 'notebook_bundle.zip',
    timestamp
  } = options;

  await mkdir(outputDir, { recursive: true });

  const {
    notebookJson,
    configYaml,
    readmeMarkdown,
    zipData
  } = await renderBundle(config, {
    createZip,
    timestamp
  });

  const notebookPath = path.join(outputDir, 'fine_tune.ipynb');
  const configPath = path.join(outputDir, 'config.yaml');
  const readmePath = path.join(outputDir, 'README.md');

  await Promise.all([
    writeFile(notebookPath, notebookJson, 'utf8'),
    writeFile(configPath, configYaml, 'utf8'),
    writeFile(readmePath, readmeMarkdown, 'utf8')
  ]);

  let zipPath: string | undefined;
  if (createZip && zipData) {
    zipPath = path.join(outputDir, zipFileName);
    await writeFile(zipPath, Buffer.from(zipData));
  }

  return {
    notebookPath,
    configPath,
    readmePath,
    zipPath
  };
}
