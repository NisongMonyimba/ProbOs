# ProbOS for Automotive / EV Battery Safety: ISO 26262 Positioning

## What this document is

A technical brief for automotive and EV functional-safety engineers,
explaining what ProbOS's already-validated battery thermal-runaway
model demonstrates, and how it relates to ISO 26262's Hazard Analysis
and Risk Assessment (HARA) process. Every technical claim below is
either (a) an actual, already-computed ProbOS result, or (b) an
accurately cited ISO 26262 concept -- nothing here is invented for
this audience.

## What ProbOS does NOT claim

Before anything else: ProbOS does not perform HARA, does not assign
ASIL (Automotive Safety Integrity Level) classifications, and has not
been used in an actual ISO 26262 safety case submission. HARA is a
structured engineering-judgment process (ISO 26262 Part 3, Clause 7)
that rates hazardous events on Severity (S0-S3), Exposure (E0-E4),
and Controllability, then derives Safety Goals and ASIL levels. That
process, and the engineering judgment it requires, remains entirely
with your team.

## What ProbOS DOES provide: quantitative evidence for HARA's inputs

HARA's Severity and Exposure ratings are ultimately judgment calls --
but judgment calls are stronger when backed by quantitative evidence
of how bad a failure could be, and how likely, rather than by
intuition or a single deterministic simulation run at nominal
parameter values.

ProbOS's BatteryModel2Cell (an 8-state Arrhenius thermal-runaway
model with 15 uncertain parameters -- SEI decomposition, anode and
cathode side reactions, per cell) is validated against Kim et al.
(2007) accelerating rate calorimetry (ARC) data. Three specific,
already-computed results are directly relevant to the kind of
evidence a HARA process needs:

1. Sobol sensitivity identifies which physical parameter actually
   drives thermal-runaway risk. The SEI decomposition activation
   energy (Ea_SEI) accounts for 45.7% of output variance
   (S1 = 0.457) -- by far the single largest driver. This is the kind
   of evidence that can inform which manufacturing or material
   specifications most affect a hazard's real-world exposure
   probability, rather than treating all 15 uncertain parameters as
   equally important.

2. Monte Carlo propagation reveals tail risk invisible to a
   deterministic simulation. Running the model across the full range
   of physically plausible parameter values (not just their
   literature-mean point estimate) shows the P95 worst-case cell
   decomposes 11.6x faster than the P50 median case. A deterministic
   safety analysis using only mean parameter values would never
   surface this -- exactly the kind of gap HARA's Severity/Exposure
   framework is meant to catch, but which requires quantitative tail
   analysis to actually see.

3. A provenance tracker traces any dangerous simulated outcome back
   to the specific parameter values that produced it. For any
   particularly severe simulated event, ProbOS can show exactly which
   combination of uncertain physical parameters caused it -- a
   documented, auditable trail from outcome to cause, directly
   relevant to the kind of evidence a safety case documentation
   package needs to support.

## Why this matters for EV battery management specifically

ISO 26262's HARA is performed early in the safety lifecycle (concept
phase), and battery thermal-runaway is exactly the kind of
malfunction-driven hazardous event HARA is designed to identify and
classify. The severity of an EV battery thermal-runaway event is
typically high (potential for fire, S2-S3 range); the exposure
probability depends heavily on manufacturing and material variability
that a single deterministic simulation cannot characterize. This is
precisely where quantitative uncertainty propagation adds evidence a
point-estimate simulation cannot provide.

## What we're asking for

Not a sale, and not a claim that this tool is ISO 26262-certified or
ready to be dropped into your safety case as-is. We're looking for a
conversation with a practicing battery or EV functional-safety
engineer to confirm (or correct) whether this kind of quantitative
uncertainty evidence would actually be useful input to your HARA
process, and if so, what would need to change for it to fit your
actual workflow.

## Technical references

- Kim, G.-H. et al. (2007) -- battery thermal-runaway validation data
- ISO 26262 Part 3, Clause 7 -- HARA process definition
- Full ProbOS technical documentation: see the project README and
  docs/vision/main.tex for the complete, honestly-labeled roadmap of
  validated versus aspirational capabilities.
