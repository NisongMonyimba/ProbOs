# Automotive / EV Safety Engineer Outreach -- Email Template

**Subject:** Quick question about uncertainty quantification in your
battery safety analysis

---

Hi [Name],

I'm building ProbOS, a probabilistic simulation tool that propagates
real parameter uncertainty (material properties, manufacturing
variance) through physical models, rather than running a single
deterministic simulation at nominal values.

I've validated it against published battery thermal-runaway data
(Kim et al. 2007) and found that the worst-case (P95) simulated cell
decomposes over 11x faster than the median case -- a tail risk a
standard deterministic simulation using mean parameter values would
never surface.

I'm not trying to sell you anything right now. I'm trying to learn
whether this kind of quantitative tail-risk evidence would actually
be useful as input to a HARA-style hazard assessment, or whether I'm
missing something about how battery safety analysis actually works in
practice at your organization.

Would you be open to a short call (15-20 minutes) to look at what
I've built and tell me honestly whether it solves a real problem or
not? I'd genuinely rather hear "this doesn't help because X" now than
keep building in the wrong direction.

Thanks for considering it either way.

P.S. -- if you want to see what I mean, here's a 30-second
interactive demo: https://nisongmonyimba.github.io/ProbOs/ -- no install, no meeting, just a click.

[Your name]
[Contact info]

---

**Notes for use (not part of the email itself):**
- Keep the ask modest and specific (15-20 minutes), not an open-ended
  sales pitch.
- Do not claim ISO 26262 certification, compliance, or prior use in
  an actual safety case -- none of these are true yet.
- The goal of this outreach is validation, not a sale.
