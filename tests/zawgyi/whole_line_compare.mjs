// Comparison only. The product path is convertLine, not this threshold.
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const engine = process.argv[2];
const { convertLine } = await import(pathToFileURL(path.join(engine, "src/convert.mjs")).href);
const { zawgyiProbability } = await import(
  pathToFileURL(path.join(engine, "src/detector.mjs")).href
);
const { zawgyiToUnicode } = await import(
  pathToFileURL(path.join(engine, "src/transliterate.mjs")).href
);

const payload = JSON.parse(fs.readFileSync(0, "utf8"));
const threshold = payload.threshold;
const out = payload.lines.map((line) => {
  const span = convertLine(line);
  const p = zawgyiProbability(line);
  const whole = Number.isFinite(p) && p > threshold ? zawgyiToUnicode(line) : line;
  return { text: span.text, words: span.words, label: span.label, whole };
});
process.stdout.write(JSON.stringify(out));
