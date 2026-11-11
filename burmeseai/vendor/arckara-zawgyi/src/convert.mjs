/*
 * Converts one line of Burmese that may be Unicode, Zawgyi, or both at once.
 *
 * Why not the usual "detect the line, then convert the line": on the benchmark in bench/, that
 * restores 1% of lines where a run of Zawgyi sits inside Unicode (the Unicode majority outvotes the
 * Zawgyi), and converting every line without detecting breaks every clean Unicode line.
 *
 * So the line is split into words at whitespace, each word gets its own evidence from the
 * detector's model, and a two state Viterbi pass picks the encoding of every word. Switching
 * encoding between neighbouring words costs `switchCost`, which keeps an ambiguous word (one
 * spelled the same way in both encodings) with its neighbours instead of flipping on noise.
 * Only runs labelled Zawgyi are converted, with Google's rules, whole runs at a time so the rules
 * see the spaces they expect.
 *
 * A word can also be half converted: Zawgyi glyphs left inside otherwise Unicode spelling, the
 * output of a converter that stopped early (ႏွင့်, တယ္, တို႔), or Unicode letters typed in visual
 * order (ြခ). Converting such a word whole would break its Unicode half, so repairWord() tries
 * small edits around each Zawgyi pattern and keeps one only if the pattern goes away (and, for a
 * partial conversion, the model reads the word as more Unicode than before).
 */
import { delta as rawDelta, zawgyiProbability } from "./detector.mjs";
import { MYANMAR, SIGNATURES, UNICODE_ONLY, signaturesIn } from "./signatures.mjs";
import { zawgyiToUnicode } from "./transliterate.mjs";

/*
 * Evidence is read off a copy with the digit ၀ inside a word taken as the letter ဝ, which is how
 * people type it in both encodings. Without this the correct conversion of Zawgyi ျမ၀တီ (Myawaddy)
 * scored lower than a wrong repair (found in native speaker review). Only the SCORING copy
 * changes; the text keeps its ၀, which native readers treat as normal.
 */
const LETTER = "[\\u1000-\\u103F\\u104C-\\u109F]";
const DIGIT_AS_WA = new RegExp(`(?<=${LETTER})\\u1040|\\u1040(?=${LETTER})`, "g");
const delta = (t) => rawDelta(t.replace(DIGIT_AS_WA, "\u101D"));

/*
 * Zero width spaces inside a word are ignored while deciding (they split ေတာ် away from its
 * consonant in visual order text), but a word keeps them unless it actually gets repaired:
 * Burmese Wikipedia uses them as word breaks, and clean Unicode must come out byte for byte.
 */
const ZWSP = /\u200B/g;

/** An asat right before a vowel sign is never valid Unicode (Zawgyi typo သမ္ိုင္း became သမ်ိုင်း). */
const ASAT_BEFORE_VOWEL = /\u103A(?=[\u102B-\u1030\u1032])/g;

// Tuned on an earlier set of benchmark cases, built from mostly different sentences: switch cost 6
// was the lowest that left every clean Unicode case untouched; searching repairs down to d = -10
// lifted half converted words by about ten points for under 1.5 points on the others, and searching
// further cost more than it gained.
export const DEFAULTS = { switchCost: 6, patternWeight: 20, strongZawgyi: -10 };

/**
 * How strongly a word should be kept (positive) or converted (negative).
 *
 * Most words are judged on the INPUT, by the detector's model: that is reliable for whole Unicode
 * or whole Zawgyi words (tested 17 Sep: judging every word by its converted output instead let
 * clean Unicode fall from 100% to 87.5%, because some garbled conversions still read as plausible).
 *
 * The exception is a word that reads as mostly Unicode yet carries a Zawgyi pattern: the half
 * converted word. For that one the two OUTPUTS are compared, the small repair against a whole
 * conversion, since converting it whole would break its Unicode half.
 */
function readings(word, patternWeight, strongZawgyi) {
  const d = delta(word.replace(ZWSP, ""));
  const patterns = patternCount(word.replace(ZWSP, ""));
  if (patterns === 0) return { keep: word, lean: UNICODE_ONLY.test(word) ? d + patternWeight : d };
  if (d < strongZawgyi) return { keep: null, lean: d - patternWeight }; // clearly Zawgyi, skip the repair search
  const score = (t) => delta(t) - patternWeight * patternCount(t);
  const keep = repairWord(word);
  return { keep, lean: score(keep) - score(zawgyiToUnicode(word)) };
}

const GLOBAL = Object.values(SIGNATURES).map((rx) => new RegExp(rx.source, "g"));

/**
 * Unicode letters stored in the order they are drawn. Only the two marks drawn LEFT of their
 * consonant get typed first: ြ (U+103C) and the e vowel. U+103B before a consonant is Zawgyi's
 * medial ra, not misplaced Unicode, and reordering it broke ျမ၀တီ into မျ၀တီ.
 */
const visualOrder = (s) =>
  s.replace(/\u103C([\u1000-\u1021])/g, "$1\u103C").replace(/\u1031([\u1000-\u1021][\u103B-\u103E]*)/g, "$1\u1031");

const patternCount = (s) => GLOBAL.reduce((n, rx) => n + (s.match(rx)?.length ?? 0), 0);

/** Lexicographic order on equal length number lists: the first difference decides. */
function ranksBefore(a, b) {
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] < b[i];
  return false;
}

/** How many code points two strings differ by, ignoring their common prefix and suffix. */
function editSize(a, b) {
  let p = 0;
  while (p < a.length && p < b.length && a[p] === b[p]) p++;
  let s = 0;
  while (s < a.length - p && s < b.length - p && a[a.length - 1 - s] === b[b.length - 1 - s]) s++;
  return Math.max(a.length, b.length) - p - s;
}

/**
 * Fixes the Zawgyi part of a mostly Unicode word, or returns it unchanged.
 *
 * Candidates are ranked: fewest patterns left, then a visual order fix before a Zawgyi
 * conversion, then the smallest edit, and only then the model's evidence. The order matters
 * because two valid words can both remove a pattern (ြဖင့် could become ဖြင့် by reordering or ဖွင့် by
 * converting), and ြ before its consonant can only be visual order: Zawgyi's letter at that code
 * point is typed after the consonant. Ranking evidence first picked the wrong word.
 */
export function repairWord(word) {
  // a virama right before a zero width space can only be a mistyped asat: stacking cannot cross
  // the space, and once the space is gone it would pass for valid stacking
  let current = word.replace(/\u1039(?=\u200B)/g, "\u103A").replace(ZWSP, "");
  for (let round = 0; round < 4; round++) {
    const before = patternCount(current);
    if (before === 0) break;
    const positions = new Set();
    for (const rx of GLOBAL) for (const m of current.matchAll(rx)) positions.add(m.index);
    let best = null;
    const baseScore = delta(current);
    for (const i of positions) {
      for (let a = 0; a <= 3; a++) {
        for (let b = 1; b <= 4; b++) {
          const s = Math.max(0, i - a), e = Math.min(current.length, i + b);
          const window = current.slice(s, e);
          [visualOrder, zawgyiToUnicode].forEach((fix, kind) => {
            const t = fix(window);
            if (t === window) return;
            const cand = current.slice(0, s) + t + current.slice(e);
            const left = patternCount(cand);
            if (left >= before) return;
            // A reorder of ြ or ေ out of a position it can never hold needs no second opinion. A
            // Zawgyi conversion must also read as more Unicode, and there the model outranks
            // edit size: after Zawgyi ႏ the ွ is Zawgyi too, so ႏွ -> နှ is right even though
            // converting ႏ alone (နွ) is the smaller edit.
            if (kind === 1 && delta(cand) <= baseScore) return;
            const rank = kind === 0 ? [left, kind, editSize(current, cand), -delta(cand)] : [left, kind, -delta(cand), editSize(current, cand)];
            if (!best || ranksBefore(rank, best.rank)) best = { cand, rank };
          });
        }
      }
    }
    if (!best) break;
    current = best.cand;
  }
  // A word that needed a visual order repair was typed by someone with that habit, so a ြ still
  // sitting before a consonant is tried as well, but only after a letter that never takes medial
  // ra (အြမင်). ဖြင့် is valid, and moving its ြ on the model's say-so produced ဖငြ့်.
  if (current !== word.replace(ZWSP, "")) {
    for (const m of [...current.matchAll(/(?<=[\u1021\u101A-\u1020\u1025-\u102A])\u103C(?=[\u1000-\u1021])/g)]) {
      const cand = current.slice(0, m.index) + current[m.index + 1] + "\u103C" + current.slice(m.index + 2);
      if (delta(cand) > delta(current)) current = cand;
    }
  }
  // unchanged apart from zero width spaces: give the word back exactly as it came in
  return current === word.replace(ZWSP, "") ? word : current;
}

/*
 * Karen and Shan written correctly in Unicode share code points with Zawgyi's glyph variants, the
 * detector's model reads them as Zawgyi, and converting them destroys them (found in CC-100 on
 * 17 Sep 2026, e.g. S'gaw Karen တၢ်ကွဲးဖှိၣ်). Their commonest syllable endings are Karen ၢ် ၣ်, a Shan
 * consonant with asat, and a Shan tone after asat. In a 55,675 line sample of CC-100's Zawgyi
 * file, every line with two of them was Karen or Shan, so such a line is left exactly as it is.
 */
const OTHER_LANGUAGE = /[\u1062\u1063]\u103A|[\u1075-\u1081]\u103A|\u103A[\u1087-\u108A]|[\u105B-\u105F]/g;

/*
 * Lines no conversion can make readable, left untouched and labelled so a corpus can drop them.
 * Measured 17 Sep 2026 as lines with two or more hits: 0 of 2,500 clean Wikipedia sentences, and
 * the hits in CC-100 were what they claim to be.
 *   unreadable: a consonant followed directly by visarga. Not Zawgyi but an older font that uses း
 *     where Unicode has an asat (အခိနးကို ရောကးလာခဲံလေပြီ၈); converting it only moves the garbage.
 *   damaged: medial ya after a letter that never takes it. Unicode that already went through a
 *     Zawgyi converter before it was crawled (အသကျ for အသက်, နှစျ for နှစ်): 83 lines in a 17,011
 *     line sample of CC-100's Unicode file. Checked on OUTPUT, because in Zawgyi input the same
 *     code point is a medial ra typed before the next consonant (ေနျပည္ေတာ္).
 */
const LEGACY_FONT = /[\u1000-\u1021]\u1038/g;
const DOUBLE_CONVERTED = /[\u1004\u1010\u1014\u100A\u101A]\u103B(?![\u102B-\u1032\u1036-\u103E])/g;
const hits = (text, rx) => text.match(rx)?.length ?? 0;

/**
 * Upper vowel before lower vowel (ို not ုိ), the order Unicode text is stored and searched in.
 * Anusvara is NOT an upper vowel here: ုံ is already the standard order, and including it turned
 * every လုံး into လံုး (caught by the benchmark on 17 Sep, invisible in the CC-100 statistics).
 */
const LOWER_BEFORE_UPPER = /([\u102F\u1030])([\u102D\u102E\u1032])/g;

/**
 * Converts one line. Clean Unicode comes back byte for byte; lines it cannot make readable come
 * back unchanged with a label saying why.
 *
 * @param {string} line
 * @param {{ switchCost?: number, patternWeight?: number, strongZawgyi?: number }} [opts]
 * @returns {{ text: string,
 *   label: "none"|"unicode"|"zawgyi"|"mixed"|"repaired"|"other-language"|"unreadable"|"damaged",
 *   zawgyiWords: number, repaired: number, words: number, flags: string[] }}
 */
export function convertLine(line, opts = {}) {
  const { switchCost, patternWeight, strongZawgyi } = { ...DEFAULTS, ...opts };
  if (!MYANMAR.test(line)) return { text: line, label: "none", zawgyiWords: 0, repaired: 0, words: 0, flags: [] };
  const untouched = (label) => ({ text: line, label, zawgyiWords: 0, repaired: 0, words: 0, flags: [label] });
  if (hits(line, OTHER_LANGUAGE) >= 2) return untouched("other-language");
  // LEGACY_FONT is only ever read off the OUTPUT: in Zawgyi input the e vowel precedes its
  // consonant, so consonant + visarga (ေအး, ေပး) is normal there. Checking input too marked 7% of
  // CC-100's Zawgyi file unreadable and dropped the benchmark's whole Zawgyi lines to 63% (17 Sep).

  // words and the whitespace between them, in order
  const parts = line.split(/(\s+)/);
  const words = []; // indices into parts of words that carry Myanmar text
  for (let i = 0; i < parts.length; i += 2) if (MYANMAR.test(parts[i])) words.push(i);

  // Viterbi over the Myanmar words: state U keeps a word, state Z converts it.
  const n = words.length;
  const r = words.map((i) => readings(parts[i], patternWeight, strongZawgyi));
  const vU = new Float64Array(n), vZ = new Float64Array(n);
  const fromU = new Uint8Array(n), fromZ = new Uint8Array(n); // 1 when the best path came from the other state
  for (let k = 0; k < n; k++) {
    const eU = r[k].lean / 2, eZ = -r[k].lean / 2;
    if (k === 0) { vU[0] = eU; vZ[0] = eZ; continue; }
    const stayU = vU[k - 1], swapU = vZ[k - 1] - switchCost;
    const stayZ = vZ[k - 1], swapZ = vU[k - 1] - switchCost;
    if (stayU >= swapU) { vU[k] = stayU + eU; fromU[k] = 0; } else { vU[k] = swapU + eU; fromU[k] = 1; }
    if (stayZ >= swapZ) { vZ[k] = stayZ + eZ; fromZ[k] = 0; } else { vZ[k] = swapZ + eZ; fromZ[k] = 1; }
  }
  const isZ = new Uint8Array(n);
  let state = vZ[n - 1] > vU[n - 1] ? 1 : 0;
  for (let k = n - 1; k >= 0; k--) {
    isZ[k] = state;
    if (state === 1 ? fromZ[k] : fromU[k]) state = 1 - state;
  }

  // Kept words take their repaired spelling, if the repair changed anything.
  let repaired = 0;
  for (let k = 0; k < n; k++) {
    if (isZ[k]) continue;
    const word = parts[words[k]];
    const keep = r[k].keep ?? repairWord(word); // a Zawgyi looking word its neighbours kept
    if (keep === word) continue;
    parts[words[k]] = keep;
    repaired++;
  }

  // Convert each maximal run of Zawgyi words, including the spaces and non Myanmar words inside it.
  const out = [];
  let zawgyiWords = 0;
  let p = 0;
  let k = 0;
  while (k < n) {
    if (!isZ[k]) { k++; continue; }
    let e = k;
    while (e + 1 < n && isZ[e + 1]) e++;
    const from = words[k], to = words[e];
    out.push(parts.slice(p, from).join(""));
    const run = parts.slice(from, to + 1).join("");
    const converted = zawgyiToUnicode(run);
    out.push(converted);
    p = to + 1;
    // a run spelled the same in both encodings changed nothing, so it does not make the line Zawgyi
    if (converted !== run) zawgyiWords += e - k + 1;
    k = e + 1;
  }
  out.push(parts.slice(p).join(""));
  // only text this pipeline wrote is cleaned; clean input stays untouched byte for byte
  const text = zawgyiWords || repaired
    ? out.join("").replace(ASAT_BEFORE_VOWEL, "").replace(LOWER_BEFORE_UPPER, "$2$1")
    : out.join("");
  if (hits(text, DOUBLE_CONVERTED) >= 2 || hits(text, LEGACY_FONT) >= 2) {
    return untouched(hits(text, LEGACY_FONT) >= 2 ? "unreadable" : "damaged");
  }

  const label = zawgyiWords === 0 ? (repaired ? "repaired" : "unicode") : zawgyiWords === n ? "zawgyi" : "mixed";
  const flags = signaturesIn(text).map((s) => `residual:${s}`);
  if (zawgyiProbability(text) > 0.95) flags.push("output-still-looks-zawgyi");
  return { text, label, zawgyiWords, repaired, words: n, flags };
}
