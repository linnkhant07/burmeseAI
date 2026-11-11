/*
 * Zawgyi <-> Unicode transliteration with the CLDR rules (my-t-my-s0-zawgyi and
 * my-t-my-d0-zawgyi) as generated for Google's myanmar-tools (Apache License 2.0, Copyright 2017
 * Google LLC; see third_party/myanmar-tools/LICENSE.md).
 *
 * The rule files are loaded untouched and runPhase is their algorithm line for line, including
 * the quirk that after a rule fires the remaining rules are still tried at the new position. The
 * point is to reproduce their output exactly (checked against their own test documents in
 * bench/sanity.mjs), so any accuracy gain in this project comes from deciding WHAT to convert,
 * never from changing HOW.
 */
import fs from "node:fs";
import vm from "node:vm";

function loadRules(file, fn) {
  const ctx = { RegExp };
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(new URL(`../third_party/myanmar-tools/${file}`, import.meta.url), "utf8"), ctx);
  return vm.runInContext(`${fn}()`, ctx);
}

const Z2U = loadRules("Z2U.js", "getAllRulesZ2U");
const U2Z = loadRules("U2Z.js", "getAllRulesU2Z");

function runPhase(rules, input) {
  let out = "";
  let mid = input;
  let startOfString = true;
  while (mid.length > 0) {
    let found = false;
    for (const rule of rules) {
      if (rule.matchOnStart == null || startOfString) {
        const m = mid.match(rule.p);
        if (m != null) {
          found = true;
          const rightPartSize = mid.length - m[0].length;
          mid = mid.replace(rule.p, rule.s);
          const newStart = mid.length - rightPartSize;
          if (rule.revisit == null) {
            out += mid.substring(0, newStart);
            mid = mid.substring(newStart);
          }
        }
      }
    }
    if (!found) {
      out += mid[0];
      mid = mid.substring(1);
    }
    startOfString = false;
  }
  return out;
}

export const zawgyiToUnicode = (text) => Z2U.reduce((acc, rules) => runPhase(rules, acc), text);
export const unicodeToZawgyi = (text) => U2Z.reduce((acc, rules) => runPhase(rules, acc), text);
