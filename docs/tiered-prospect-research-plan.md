# Tiered prospect research adaptation plan

Status: staged implementation in progress. The current CLI/UI wiring exposes
an offline `tiered-preview` surface for directive and query-shape validation;
full tiered workflow execution, artifacts, browser capture, and enrichment remain
future staged work.

## Goal

Adapt the deep research workflow into a tiered prospecting system that starts
from structured search directives, discovers target companies, identifies
relevant people at each company, and gathers person-level context for
personalized outreach.

The desired output is a cited prospect package that separates:

1. company fit evidence,
2. contact/role evidence, and
3. personalization evidence.

## Search directive input contract

A run should begin with an explicit prospect-search directive rather than a
single freeform query.

Required fields:

- `industry`: target vertical such as `HVAC`, `dental`, `mortgage lender`,
  `med spa`, `home services`, or another operator-selected category.
- `target_prospect_count`: number of companies to return after filtering and
  deduplication.
- `geographic_area`: state, metro, city, county, or locality such as `Texas`,
  `DFW area`, `Dallas County`, `Phoenix metro`, or `Tampa Bay`.
- `research_criteria`: open prompt for additional constraints, examples:
  - ideal company size,
  - multi-location vs. single-location,
  - recent hiring/growth signals,
  - likely need for reactivation, lead generation, or marketing support,
  - exclusions such as franchises, aggregators, or companies already in CRM.

Optional fields:

- `negative_criteria`: disqualifying signals.
- `preferred_contact_roles`: e.g. owner, founder, CEO, practice manager,
  marketing director, growth lead, branch manager.
- `source_preferences`: maps, directories, official sites, LinkedIn, trade
  associations, local rankings, review sites, state registries, news.
- `evidence_depth`: fast, standard, or deep.
- `browser_capture`: none, screenshots only, text extraction, or full page audit.

Example directive:

```json
{
  "industry": "dental",
  "target_prospect_count": 25,
  "geographic_area": "DFW area",
  "research_criteria": "Prioritize multi-location practices or DSOs with signs of patient reactivation opportunity, paid ads, growth, or weak review follow-up. Exclude national chains.",
  "preferred_contact_roles": ["owner", "founder", "practice manager", "marketing director"],
  "evidence_depth": "standard",
  "browser_capture": "text extraction"
}
```

## Tier 1: company discovery and qualification

Purpose: identify companies that match the directive before searching for
individual contacts.

Inputs:

- Full search directive.
- Provider configuration and search max-results.
- Existing checkpoint/thread id.

Process:

1. Expand the directive into multiple company-discovery queries, for example:
   - `<industry> companies in <geographic_area>`
   - `best <industry> <geographic_area>`
   - `<industry> owner <geographic_area>`
   - `<industry> marketing <geographic_area>`
   - trade/directory/local association queries relevant to the vertical.
2. Search across configured providers.
3. Normalize evidence as snippets vs. page reads.
4. Deduplicate companies by domain, name, phone/address when available.
5. Score each company against the directive:
   - industry match,
   - geography match,
   - evidence quality,
   - fit to open criteria,
   - disqualification flags,
   - confidence.
6. Keep a ranked candidate pool larger than the requested count, then select the
   top `target_prospect_count` after deduplication and confidence filtering.

Expected Tier 1 output per company:

```json
{
  "company_id": "company_001",
  "name": "Example Dental Group",
  "website": "https://example.com",
  "industry": "dental",
  "geographic_area": "DFW area",
  "locations": ["Dallas, TX", "Plano, TX"],
  "fit_score": 0.82,
  "fit_rationale": "Matches dental + DFW, appears multi-location, has growth/marketing signals.",
  "disqualification_flags": [],
  "evidence_ids": ["ev_company_001", "ev_company_002"],
  "source_confidence": "medium"
}
```

Tier 1 stopping condition:

- Enough qualified companies are found to satisfy `target_prospect_count`, or
- the search exhausts configured attempts and returns a partial result with
  warnings.

## Tier 2: contact discovery inside each qualified company

Purpose: for each company selected in Tier 1, identify likely individuals to
contact.

Inputs:

- Qualified company list from Tier 1.
- Preferred roles from the directive.
- Company website/domain and evidence catalog.

Process per company:

1. Generate person-discovery queries scoped to the company:
   - `<company> owner`
   - `<company> founder`
   - `<company> CEO`
   - `<company> marketing director`
   - `<company> practice manager`
   - `site:<company_domain> team`
   - `site:<company_domain> about`
   - `site:linkedin.com/in <company> <role>`
2. Search web providers first for broad discovery.
3. Use browser/page capture where the company site likely contains structured
   staff, about, leadership, location, or contact pages.
4. Extract candidate contacts and normalize names, roles, source URLs, and
   confidence.
5. Rank contacts by role relevance and evidence strength.
6. Keep multiple contacts per company when appropriate, rather than forcing a
   single best contact too early.

Expected Tier 2 output per contact:

```json
{
  "contact_id": "contact_001",
  "company_id": "company_001",
  "name": "Jane Smith",
  "title": "Practice Manager",
  "role_category": "operator",
  "profile_urls": ["https://example.com/team/jane-smith"],
  "email": null,
  "phone": null,
  "contact_confidence": 0.74,
  "evidence_ids": ["ev_contact_001"],
  "notes": "Listed on company team page as practice manager."
}
```

Tier 2 stopping condition:

- Each qualified company has at least one contact, or
- the system records a `no_contact_found` warning and preserves the company as a
  company-only prospect if it still has strong fit.

## Tier 3: person-level personalization research

Purpose: gather additional context on each target person so outreach can be
specific, relevant, and evidence-backed.

Inputs:

- Contact records from Tier 2.
- Company records from Tier 1.
- Original research criteria.

Process per person:

1. Generate person-context queries:
   - `<person> <company>`
   - `<person> <company> interview`
   - `<person> <company> news`
   - `<person> <company> LinkedIn`
   - `<person> <industry> <geographic_area>`
2. Search for public context tied to the person and company.
3. Use browser/page capture for pages that search snippets cannot represent
   reliably, such as:
   - profile pages,
   - team pages,
   - local business features,
   - event/speaker pages,
   - company blog posts,
   - press releases,
   - public review or testimonial pages.
4. Extract personalization signals:
   - role responsibilities,
   - recent company initiatives,
   - hiring/growth signals,
   - service expansion,
   - review/reputation signals,
   - local community involvement,
   - likely pain points mapped to the campaign offer.
5. Produce outreach angles that cite evidence and distinguish direct facts from
   inferred messaging opportunities.

Expected Tier 3 output per person:

```json
{
  "contact_id": "contact_001",
  "personalization_signals": [
    {
      "signal": "Practice recently opened a second DFW location.",
      "message_angle": "Lead with patient reactivation and recall campaigns for multi-location growth.",
      "evidence_ids": ["ev_person_001"],
      "confidence": "medium"
    }
  ],
  "suggested_opening_line": "Saw that Example Dental Group has expanded in the DFW area; teams at that stage often need a reliable way to reactivate dormant patient lists across locations.",
  "do_not_claim": [
    "Do not claim direct responsibility for marketing unless a source confirms it."
  ]
}
```

Tier 3 stopping condition:

- Each selected contact has at least one usable personalization signal, or
- the system flags the contact as needing human review before outreach.

## Where Playwright or an agent browser fits

Use Playwright/agent-browser capture selectively. Search APIs are efficient for
candidate discovery, but browser automation is valuable when pages need direct
inspection, rendering, or structured extraction.

Recommended browser use cases:

- Company website validation:
  - confirm the official domain,
  - capture homepage/about/location pages,
  - verify geography and services,
  - extract multi-location or specialty claims.
- Team/contact discovery:
  - inspect team, about, leadership, contact, providers, doctors, agents, loan
    officers, or staff pages,
  - handle JavaScript-rendered staff cards,
  - capture source screenshots for review.
- Person-level personalization:
  - inspect public profile pages and company bios,
  - extract quotes or role descriptions only when visible on public pages,
  - capture evidence snapshots for later audit.
- Quality control:
  - screenshot final source pages used for high-confidence prospects,
  - preserve rendered text when provider snippets are incomplete.

Avoid browser use for:

- Every search result by default; this will be slow and noisy.
- Sites with login walls, private data, or bot-sensitive flows.
- Any action that submits forms, sends messages, downloads private files, or
  mutates external systems.

Suggested browser artifact fields:

```json
{
  "browser_capture_id": "cap_001",
  "url": "https://example.com/team",
  "company_id": "company_001",
  "contact_id": "contact_001",
  "captured_at": "ISO-8601 timestamp",
  "text_excerpt": "Visible page text used as evidence...",
  "screenshot_path": "artifacts/thread-id/captures/cap_001.png",
  "dom_snapshot_path": "artifacts/thread-id/captures/cap_001.html",
  "evidence_ids": ["ev_contact_001"]
}
```

## Proposed workflow shape

The current local workflow can be adapted from one generic `researcher` loop into
explicit tier nodes:

```text
main
  -> directive_parser
  -> company_discovery
  -> company_qualification
  -> contact_discovery
  -> contact_enrichment
  -> personalization_research
  -> artifact_writer
  -> review
```

Suggested node responsibilities:

- `directive_parser`: validate industry, count, geography, and open criteria.
- `company_discovery`: run Tier 1 search queries and collect raw company
  candidates.
- `company_qualification`: dedupe, score, and select companies.
- `contact_discovery`: run Tier 2 company-scoped person discovery.
- `contact_enrichment`: normalize contacts, roles, and confidence.
- `personalization_research`: run Tier 3 person-scoped research.
- `browser_capture`: optional helper node called by company/contact/person nodes
  when rendered pages are likely to improve evidence quality.
- `artifact_writer`: write JSON/CSV/Markdown outputs with company, contact, and
  personalization sections.
- `review`: interrupt for human approval before any outreach use.

## Data model additions to plan for

Current artifacts already support evidence and prospect records. A later
implementation should add explicit nested records instead of overloading one flat
prospect object.

Recommended entities:

- `SearchDirective`
- `CompanyProspect`
- `ContactCandidate`
- `PersonalizationSignal`
- `BrowserCapture`
- `TieredResearchRun`

Recommended IDs:

- `company_id`: stable within a run, derived from normalized company/domain.
- `contact_id`: stable within a run, derived from company + person name + role.
- `evidence_id`: stable citation target for snippet, page-read, or browser
  capture evidence.
- `capture_id`: stable browser-capture artifact reference.

## Artifact output plan

Each run should write:

- `tiered_prospect_research.json`: full normalized data model.
- `companies.csv`: one row per qualified company.
- `contacts.csv`: one row per contact candidate.
- `personalization.csv`: one row per personalization signal.
- `research_report.md`: human-readable report grouped by company and contact.
- `captures/`: optional screenshots, text snapshots, and DOM snapshots.

Markdown report structure:

```text
# Tiered Prospect Research Report

## Directive
## Summary
## Qualified Companies
### Company: <name>
#### Fit Evidence
#### Contact Candidates
##### Contact: <name>, <role>
###### Personalization Signals
###### Suggested Outreach Angle
###### Evidence
## Warnings and Human Review Items
```

## Review and safety gates

The workflow should keep a human review interrupt before any outreach, export to
CRM, or message generation that could be sent externally.

Review should check:

- Does the company actually match the industry and geography?
- Is the contact currently associated with the company?
- Are role/title claims cited?
- Are personalization statements grounded in evidence?
- Are inferred claims clearly labeled as inferred?
- Are sources public and appropriate to use?
- Are no private/login-only sources included?

## Testing strategy for later implementation

Keep the implementation offline-testable.

Recommended tests:

- Directive parsing validates required fields and preserves freeform criteria.
- Query generation includes industry, geography, and role terms.
- Tier 1 dedupes companies by normalized domain/name.
- Tier 1 respects `target_prospect_count` after qualification.
- Tier 2 scopes person search to each selected company.
- Tier 3 stores personalization signals with evidence IDs.
- Browser capture helper can be mocked; no live browser or network in unit tests.
- Artifact writers produce companies, contacts, personalization, and markdown
  sections without secrets.
- UI/CLI preview renders directive, tier progress, warnings, and artifact paths.

## Implementation notes for later

- Do not replace the existing local checkpoint workflow; extend it with tiered
  state and explicit nodes.
- Keep provider calls injectable so tests can use mocked search results.
- Keep browser capture optional and injectable so tests can use static HTML.
- Preserve snippet vs. page-read vs. browser-captured evidence semantics.
- Keep API keys environment/session-only; do not persist keys in checkpoints or
  artifacts.
- Prefer partial results with warnings over failing the whole run when one
  company or contact cannot be enriched.


## Current CLI/UI preview surface

The implemented preview entrypoint is intentionally offline and shallow:

```bash
python -m deep_research_agent tiered-preview \
  --industry "dental" \
  --geography "DFW area" \
  --criteria "multi-location practices" \
  --preferred-contact-role owner \
  --json
```

This command and the Streamlit preview helper build a `TieredSearchDirective`,
show company/contact/person query templates, and list the planned artifact names.
They do not run live search, instantiate `AsyncMultiProviderSearch()`, launch a
browser, enrich contacts, export CRM data, or generate outreach. Runtime tier
execution should keep the same injected-client boundary so early discovery can
exclude Exa until a human-approved enrichment phase.
