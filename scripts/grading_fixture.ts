// Generates tests/fixtures/grading_parity.json from the original TypeScript grading module.
// Usage: bun scripts/grading_fixture.ts ../agentic-rag-mastra > tests/fixtures/grading_parity.json
import { resolve } from "node:path";

const root = resolve(process.argv[2] ?? "../agentic-rag-mastra");
const g = await import(`${root}/src/lib/grading.ts`);

const answers = [
  "Ja, Nutzer werden informiert [chunk:art50-p1].",
  "Siehe [chunk:art50-p1] und [chunk:art5-p1].",
  "[KEINE_GRUNDLAGE] Die abgerufenen Auszüge enthalten dazu keine Grundlage.",
  "[KEINE_GRUNDLAGE] aber [chunk:art99-p3]",
  "Eine Antwort ohne Zitat.",
  "Bis zu 35.000.000 EUR oder 7 % [chunk:art99-p3] [chunk:art99-p3]",
  "Bis zu 35 000 000 EUR [chunk:art99-p3], 15.000.000 [chunk:art99-p4]",
  "Mehrfach [chunk:art50-p1, chunk:art50-p2] zitiert.",
  "Ab dem 2. August 2026 MASCHINENLESBAR [chunk:art113-p0]",
  "1.000x und 12.345ab [chunk:art4-p1]",
  "",
];
const retrieved = [["art50-p1", "art99-p3", "art113-p0", "art4-p1"], []];
const facts = [["informiert"], ["35000000|35 millionen", "7 %"], ["maschinenlesbar", "2. august 2026"], ["1000x", "12345ab"], []];
const expectedChunks = [["art50-p1"], ["art99-p3", "art99-p4"], []];

const rows = answers.map((answer) => ({
  answer,
  citations: g.parseCitations(answer),
  isRefusal: g.isRefusal(answer),
  normalized: g.normalize(answer),
  validity: retrieved.map((r) => g.citationValidity(answer, new Set(r))),
  refusal: ["answered", "refused"].map((e) => g.refusalCorrectness(e, answer)),
  recall: expectedChunks.map((c) => g.citationRecall(c, answer)),
  facts: facts.map((f) => g.factCoverage(f, answer)),
}));

console.log(JSON.stringify({ retrieved, facts, expectedChunks, rows }, null, 2));
