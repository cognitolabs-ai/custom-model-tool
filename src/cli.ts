#!/usr/bin/env node
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

import { generateNotebook } from './generateNotebook.js';

interface CliArguments {
  configPath: string;
  outputDir: string;
  skipZip: boolean;
}

function parseArguments(argv: string[]): CliArguments {
  const args = { configPath: '', outputDir: '', skipZip: false };

  for (let i = 0; i < argv.length; i += 1) {
    const current = argv[i];
    switch (current) {
      case '--config':
      case '-c':
        args.configPath = argv[++i] ?? '';
        break;
      case '--out':
      case '-o':
        args.outputDir = argv[++i] ?? '';
        break;
      case '--no-zip':
        args.skipZip = true;
        break;
      case '--help':
      case '-h':
        printUsage();
        process.exit(0);
        break;
      default:
        if (current.startsWith('-')) {
          throw new Error(`Unknown option: ${current}`);
        }
    }
  }

  if (!args.configPath) {
    throw new Error('Missing required --config <path> argument.');
  }

  if (!args.outputDir) {
    throw new Error('Missing required --out <directory> argument.');
  }

  return args;
}

function printUsage(): void {
  const message = `
CognitioLabs Notebook Generator CLI
Links: https://www.CognitioLabs.eu | https://github.com/cognitolabs-ai

Usage:
  pnpm generate --config <config.json> --out <output_dir> [--no-zip]

Options:
  -c, --config    Path to a JSON config file (required)
  -o, --out       Output directory for generated artifacts (required)
  --no-zip        Skip creating the ZIP bundle
  -h, --help      Show this help message
`.trim();
  console.log(message);
}

async function main() {
  try {
    const args = parseArguments(process.argv.slice(2));
    const resolvedConfigPath = path.resolve(process.cwd(), args.configPath);
    const resolvedOutputDir = path.resolve(process.cwd(), args.outputDir);

    const configRaw = await readFile(resolvedConfigPath, 'utf8');
    const config = JSON.parse(configRaw);

    const result = await generateNotebook(config, {
      outputDir: resolvedOutputDir,
      createZip: !args.skipZip
    });

    console.log(
      [
        'Generated artifacts:',
        `- Notebook: ${path.relative(process.cwd(), result.notebookPath)}`,
        `- Config:   ${path.relative(process.cwd(), result.configPath)}`,
        `- README:   ${path.relative(process.cwd(), result.readmePath)}`,
        result.zipPath ? `- ZIP:      ${path.relative(process.cwd(), result.zipPath)}` : null
      ]
        .filter(Boolean)
        .join('\n')
    );
  } catch (err) {
    console.error((err as Error).message);
    process.exitCode = 1;
  }
}

void main();
