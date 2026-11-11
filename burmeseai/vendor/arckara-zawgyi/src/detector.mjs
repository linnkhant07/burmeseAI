/*
 * Zawgyi detection, ported from Google's myanmar-tools (Apache License 2.0, Copyright 2017
 * Google LLC; see third_party/myanmar-tools/LICENSE.md). The model and the arithmetic are theirs.
 *
 * What is ours: `delta()` exposes the model's evidence for any slice of a line, not just one
 * probability for the whole line. The model is a Markov chain over code points that stores, for
 * every transition, log P(Unicode) minus log P(Zawgyi). Summed over a whole line that is the
 * detector's answer; summed over one word it says which encoding that word is in, which is what
 * lets a line that mixes the two be split instead of misjudged as a whole.
 */
import fs from "node:fs";

const MODEL = new URL("../third_party/myanmar-tools/zawgyiUnicodeModel.dat", import.meta.url);

const STD_CP0 = 0x1000, STD_CP1 = 0x103f;
const AFT_CP0 = 0x104a, AFT_CP1 = 0x109f;
const EXA_CP0 = 0xaa60, EXA_CP1 = 0xaa7f;
const EXB_CP0 = 0xa9e0, EXB_CP1 = 0xa9ff;
const SPC_CP0 = 0x2000, SPC_CP1 = 0x200b;
const STD_OFFSET = 1;
const AFT_OFFSET = STD_OFFSET + STD_CP1 - STD_CP0 + 1;
const EXA_OFFSET = AFT_OFFSET + AFT_CP1 - AFT_CP0 + 1;
const EXB_OFFSET = EXA_OFFSET + EXA_CP1 - EXA_CP0 + 1;
const SPC_OFFSET = EXB_OFFSET + EXB_CP1 - EXB_CP0 + 1;

function stateOf(cp, ssv) {
  if (STD_CP0 <= cp && cp <= STD_CP1) return cp - STD_CP0 + STD_OFFSET;
  if (AFT_CP0 <= cp && cp <= AFT_CP1) return cp - AFT_CP0 + AFT_OFFSET;
  if (EXA_CP0 <= cp && cp <= EXA_CP1) return cp - EXA_CP0 + EXA_OFFSET;
  if (EXB_CP0 <= cp && cp <= EXB_CP1) return cp - EXB_CP0 + EXB_OFFSET;
  if (ssv === 0 && SPC_CP0 <= cp && cp <= SPC_CP1) return cp - SPC_CP0 + SPC_OFFSET;
  return 0;
}

function load() {
  const buf = fs.readFileSync(MODEL);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let o = 0;
  const expect = (lead, trail) => {
    if (view.getUint32(o) !== lead || view.getUint32(o + 4) !== trail) throw new Error("bad model magic");
    o += 8;
  };
  expect(0x555a4d4f, 0x44454c20); // "UZMODEL "
  const version = view.getUint32(o); o += 4;
  let ssv = 0;
  if (version === 2) { ssv = view.getUint32(o); o += 4; } else if (version !== 1) throw new Error("model version");
  expect(0x424d4152, 0x4b4f5620); // "BMARKOV "
  if (view.getUint32(o) !== 0) throw new Error("markov version");
  o += 4;
  const size = view.getInt16(o); o += 2;
  const table = new Float64Array(size * size);
  for (let i1 = 0; i1 < size; i1++) {
    let entries = view.getInt16(o); o += 2;
    let fallback = 0;
    if (entries !== 0) { fallback = view.getFloat32(o); o += 4; }
    let next = -1;
    for (let i2 = 0; i2 < size; i2++) {
      if (entries > 0 && next < i2) { next = view.getInt16(o); o += 2; entries--; }
      if (next === i2) { table[i1 * size + i2] = view.getFloat32(o); o += 4; }
      else table[i1 * size + i2] = fallback;
    }
  }
  return { ssv, size, table };
}

const { ssv, size, table } = load();

/**
 * Evidence for `text`: log P(Unicode) minus log P(Zawgyi), the same sum Google's detector makes.
 * Positive leans Unicode, negative leans Zawgyi, 0 means no Myanmar signal at all.
 */
export function delta(text) {
  let prev = 0;
  let total = 0;
  for (let i = 0; i <= text.length; i++) {
    const cur = i === text.length ? 0 : stateOf(text.charCodeAt(i), ssv);
    if (prev !== 0 || cur !== 0) total += table[prev * size + cur];
    prev = cur;
  }
  return total;
}

/** Google's detector output: the probability that `text` is Zawgyi, or -Infinity with no signal. */
export function zawgyiProbability(text) {
  let prev = 0;
  let total = 0;
  let seen = false;
  for (let i = 0; i <= text.length; i++) {
    const cur = i === text.length ? 0 : stateOf(text.charCodeAt(i), ssv);
    if (prev !== 0 || cur !== 0) { total += table[prev * size + cur]; seen = true; }
    prev = cur;
  }
  return seen ? 1 / (1 + Math.exp(total)) : -Infinity;
}
