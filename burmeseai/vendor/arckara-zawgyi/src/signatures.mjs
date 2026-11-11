/*
 * Code point patterns that valid Unicode Burmese never contains and Zawgyi text regularly does.
 * They are used to CHECK output, not to decide conversion: a line that still matches one after
 * the pipeline has run is flagged for a human instead of being passed as clean.
 *
 * Learned the hard way on real data: asat followed by medial ya (ယောက်ျား, ခ်ျ) is legitimate
 * Unicode, and U+1060-U+1097 also holds real Shan, so a hit is evidence, not proof.
 */
export const SIGNATURES = {
  // U+1039 used as a visible killer; in Unicode it must be followed by a consonant.
  viramaAsAsat: /\u1039(?![\u1000-\u1021])/,
  // the e vowel typed before its consonant, at the start of a word
  eVowelFirst: /(?:^|[\s"\u201C\u2018(\[\u104B\u104A,:])\u1031(?=[\u1000-\u1021])/,
  // Zawgyi's glyph variants (ႏ ႔ ႐ ၤ ၼ ...), which in Unicode are minority-language letters
  zawgyiCodePoint: /[\u105A\u1060-\u1097]/,
  // a medial with nothing it can attach to; asat + ya is allowed, asat + ra/wa/ha is not
  medialMisplaced: /(?<![\u1000-\u1021\u103A-\u103E])[\u103B-\u103E]|(?<=\u103A)[\u103C-\u103E]/,
  // the e vowel stored before a medial
  eVowelThenMedial: /\u1031[\u103B-\u103E]/,
  // the e vowel after something it can never follow in Unicode (an asat, a vowel sign, a zero
  // width space...). Zawgyi's ္ေ and visual order Unicode's ်ေ both land here (found in native
  // speaker review: ကျွန်ေတာ် went unrepaired because nothing caught it).
  eVowelOrphan: /(?<![\u1000-\u1021\u103B-\u103E\s"\u201C\u2018(\[\u104B\u104A,:]|^)\u1031/,
};

/*
 * The reverse: orderings only Unicode produces. In Zawgyi the e vowel is typed first and is
 * always followed by its consonant, so an e vowel followed directly by a vowel sign or a final
 * (ော, နေ့) cannot be Zawgyi; nor can Unicode's three code point kinzi. Found when a Unicode
 * word between two long Zawgyi words (အောက်တိုဘာ) was converted and broken.
 */
export const UNICODE_ONLY = /\u1031[\u102B-\u1030\u1032\u1036-\u103A]|\u1004\u103A\u1039/;

const ENTRIES = Object.entries(SIGNATURES);

/** Names of the patterns `text` matches, empty when none do. */
export function signaturesIn(text) {
  const hits = [];
  for (const [name, rx] of ENTRIES) if (rx.test(text)) hits.push(name);
  return hits;
}

export const MYANMAR = /[\u1000-\u109F\uA9E0-\uA9FF\uAA60-\uAA7F]/;
